import copy
import json
import logging
import os
import shutil
from datetime import date
import getpass

import ndex2
import numpy as np
import pandas as pd
from cellmaps_utils.ndexupload import NDExHierarchyUploader
from cellmaps_utils import constants
from cellmaps_utils.provenance import ProvenanceUtil

from cellmaps_vnn.util import copy_and_register_gene2id_file
import ndex2.constants as ndexconstants

import cellmaps_vnn
import cellmaps_vnn.constants as vnnconstants
from ndex2.cx2 import RawCX2NetworkFactory, CX2Network

from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class VNNAnnotate:
    COMMAND = 'annotate'

    DEFAULT_NDEX_SERVER = 'ndexbio.org'
    DEFAULT_PASSWORD = '-'

    def __init__(self, outdir, model_predictions, disease=None, hierarchy=None, parent_network=None,
                 ndexserver=DEFAULT_NDEX_SERVER, ndexuser=None, ndexpassword=DEFAULT_PASSWORD,
                 visibility=False, slurm=False, slurm_partition=None, slurm_account=None):
        """
        Constructor. Sets up the hierarchy path either directly from the arguments or by looking for
        a hierarchy.cx2 file in the first RO-Crate directory provided. If neither is found, raises an error.

        :raises CellmapsvnnError: If no hierarchy path is specified or found.
        """
        self._outdir = os.path.abspath(outdir)
        self.original_hierarchy = None
        self.hierarchy = hierarchy
        self.parent_network = parent_network
        self._model_predictions = model_predictions
        self._disease = disease
        self._ndexserver = ndexserver
        self._ndexuser = ndexuser
        self._ndexpassword = ndexpassword
        self._visibility = visibility
        self._slurm = slurm
        self._slurm_partition = slurm_partition
        self._slurm_account = slurm_account

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'annotate' command.
        """
        # TODO: modify description later
        desc = f"""
        Version: {cellmaps_vnn.__version__}

        The 'annotate' command reads model prediction outputs (RLIPP scores) from one or more
        RO-Crate directories and uses them to annotate a given hierarchy network in CX2 format.
        The command can optionally filter scores by disease, generate SLURM scripts for batch
        processing, and upload annotated networks to NDEx for visualization and sharing.
        """
        parser = subparsers.add_parser(VNNAnnotate.COMMAND,
                                       help='Run prediction using a trained model',
                                       description=desc,
                                       formatter_class=constants.ArgParseFormatter)
        parser.add_argument('outdir', help='Directory to write results to')
        parser.add_argument('--model_predictions', nargs='+', required=True,
                            help='Path to one or multiple RO-Crate with the predictions and interpretations '
                                 'obtained from predict step',
                            type=str)
        parser.add_argument('--disease', help='Specify the disease or cancer type for which the annotations will be '
                                              'performed. This allows the annotation process to tailor the results '
                                              'according to the particular disease or cancer type. If not set, '
                                              'prediction importance scores for all diseases will be aggregated.',
                            type=str)
        parser.add_argument('--hierarchy', help='Path to hierarchy (optional). If not set, the process will search for '
                                                'hierarchy.cx2 in first RO-Crate (passed in --model_predictions).',
                            type=str)
        parser.add_argument('--parent_network', help='Path to interactome (parent network, optional) of the annotated '
                                                     'hierarchy or NDEx UUID of parent network. If not set, the '
                                                     'process will search for hierarchy_parent.cx2 in first RO-Crate '
                                                     '(passed in --model_predictions).', type=str)
        parser.add_argument('--ndexserver', default=VNNAnnotate.DEFAULT_NDEX_SERVER,
                            help='Server where annotated hierarchy will be uploaded to')
        parser.add_argument('--ndexuser',
                            help='NDEx user account. Required if uploading to NDEx.')
        parser.add_argument('--ndexpassword', default=VNNAnnotate.DEFAULT_PASSWORD,
                            help='NDEx password. Enter "-" to input password interactively, or provide a file '
                                 'containing the password. Required if uploading to NDEx.')
        parser.add_argument('--visibility', action='store_true',
                            help='If set, makes Hierarchy and interactome network loaded onto '
                                 'NDEx publicly visible')
        parser.add_argument('--slurm', help='If set, slurm script for training will be generated.',
                            action='store_true')
        parser.add_argument('--slurm_partition', help='Slurm partition', type=str)
        parser.add_argument('--slurm_account', help='Slurm account', type=str)

    def run(self):
        """
        The logic for annotating hierarchy with prediction results from cellmaps_vnn. It aggregates prediction scores
        from models, optionally filters them for a specific disease, and annotates the hierarchy with these scores.
        """
        self._process_rocrates()
        self._process_input_hierarchy_and_parent()
        hierarchy_cx, original_hierarchy_cx = self._get_hierarchy_cx()
        parent_cx = self._get_parent_cx()
        interactome_dict, root_id = self._annotate_interactomes_of_systems(parent_cx, hierarchy_cx)
        self._process_scores_and_annotate_hierarchy(hierarchy_cx, original_hierarchy_cx)
        self._upload_to_ndex_if_credentials_provided(interactome_dict, root_id)

    def register_outputs(self, outdir, description, keywords, provenance_utils):
        """
        Registers the output files of the annotation process with the FAIRSCAPE service for data provenance.
        This includes the annotated hierarchy and the RLIPP output files.

        :param outdir: The output directory where the files are stored.
        :type outdir: str
        :param description: A description of the files for provenance registration.
        :type description: str
        :param keywords: A list of keywords associated with the files.
        :type keywords: list
        :param provenance_utils: The utility class for provenance registration.
        :type provenance_utils: ProvenanceUtility
        :return: A list of dataset IDs assigned to the registered files.
        :rtype: list
        """
        hierarchy_id = self._register_hierarchy(outdir, description, keywords, provenance_utils)
        rlipp_id = self._register_rlipp_file(outdir, description, keywords, provenance_utils)
        return_ids = [hierarchy_id, rlipp_id]
        gene2ind_path = os.path.join(self._model_predictions[0], 'gene2ind.txt')
        if os.path.exists(gene2ind_path):
            gene2ind_id = copy_and_register_gene2id_file(gene2ind_path, outdir, description, keywords,
                                                         provenance_utils)
            return_ids.append(gene2ind_id)
        if self.original_hierarchy is not None:
            original_hierarchy_id = self._register_original_hierarchy(outdir, description, keywords, provenance_utils)
            return_ids.append(original_hierarchy_id)
        if self.parent_network is not None and os.path.isfile(self.parent_network):
            hierarchy_parent_id = self._copy_and_register_hierarchy_parent(outdir, description, keywords,
                                                                           provenance_utils)
            return_ids.append(hierarchy_parent_id)
        return return_ids

    def _process_scores_and_annotate_hierarchy(self, hierarchy_cx, original_hierarchy_cx):
        """
        Orchestrates the score aggregation and annotation of the hierarchy network.

        This includes:
        - Aggregating RLIPP importance scores from prediction files
        - Selecting scores for a specific disease or averaging across diseases
        - Annotating nodes with scores
        - Propagating importance to hierarchy edges
        - Writing updated networks to file

        :param hierarchy_cx: Loaded CX2 network representing the hierarchy
        :param original_hierarchy_cx: Loaded original CX2 hierarchy (if available)
        """
        self._aggregate_importance_scores_from_models()
        if self._disease is None:
            annotation_dict = self._aggregate_scores_from_diseases()
        else:
            annotation_dict = self._get_scores_for_disease(self._disease)
        if len(annotation_dict) == 0:
            print("No system importance scores available for annotation. Training was not sufficient. "
                  "Increase number of epochs and run train and predict again.")
            raise CellmapsvnnError("No system importance scores available for annotation. "
                                   "Please ensure valid data is provided for the hierarchy annotation.")

        hierarchy, original_hierarchy = self._annotate_hierarchy(annotation_dict, hierarchy_cx, original_hierarchy_cx)
        hierarchy = self._annotate_hierarchy_edges(hierarchy)
        hierarchy = self._add_provenance_info(hierarchy)
        self._save_hierarchy_to_file(hierarchy, original_hierarchy)

    def _get_rlipp_out_dest_file(self):
        """
        Constructs the file path for the RLIPP output file within the specified output directory.

        :return: The file path for the RLIPP output file.
        :rtype: str
        """
        return os.path.join(self._outdir, vnnconstants.RLIPP_OUTPUT_FILE)

    def _get_hierarchy_dest_file(self):
        """
        Constructs the file path for the hierarchy output file within the specified output directory.

        :return: The file path for the hierarchy output file.
        :rtype: str
        """
        return os.path.join(self._outdir, vnnconstants.HIERARCHY_FILENAME)

    def _get_original_hierarchy_dest_file(self):
        """
        Constructs the file path for the hierarchy output file within the specified output directory.

        :return: The file path for the hierarchy output file.
        :rtype: str
        """
        return os.path.join(self._outdir, vnnconstants.ORIGINAL_HIERARCHY_FILENAME)

    def _aggregate_importance_scores_from_models(self):
        """
        Aggregates prediction scores from multiple models' outputs by averaging them.
        The aggregated scores are then saved to the RLIPP output destination file.
        """
        data = {}

        for directory in self._model_predictions:
            filepath = os.path.join(directory, vnnconstants.RLIPP_OUTPUT_FILE)
            has_disease = False
            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith('Term') or not line.strip():
                        if 'Disease' in line:
                            has_disease = True
                        continue

                    parts = line.strip().split('\t')
                    if has_disease:
                        key = (parts[0], parts[-1])  # (Term, Disease)
                        values = np.array([float(v) for v in parts[1:-1]])
                    else:
                        key = (parts[0], 'unspecified')
                        values = np.array([float(v) for v in parts[1:]])

                    if key not in data:
                        data[key] = []
                    data[key].append(values)

        averaged_data = {k: np.mean(v, axis=0) for k, v in data.items()}

        with open(self._get_rlipp_out_dest_file(), 'w') as outfile:
            outfile.write("Term\tP_rho\tP_pval\tC_rho\tC_pval\tRLIPP\tDisease\n")
            for (term, disease), values in averaged_data.items():
                outfile.write(f"{term}\t" + "\t".join([f"{v:.5e}" for v in values]) + f"\t{disease}\n")

    def _aggregate_scores_from_diseases(self):
        """
        Aggregates the prediction scores for all diseases by averaging P_rho score.

        :return: A dictionary mapping each term to its averaged P_rho score across all diseases.
        :rtype: dict
        """
        data = pd.read_csv(self._get_rlipp_out_dest_file(), sep='\t')
        aggregated_data = data.groupby('Term').agg({
            vnnconstants.PRHO_SCORE: 'mean',
            vnnconstants.P_PVAL_SCORE: 'mean',
            vnnconstants.CRHO_SCORE: 'mean',
            vnnconstants.C_PVAL_SCORE: 'mean',
            vnnconstants.RLIPP_SCORE: 'mean'
        })

        aggregated_dict = {
            term: [row[vnnconstants.PRHO_SCORE], row[vnnconstants.P_PVAL_SCORE],
                   row[vnnconstants.CRHO_SCORE], row[vnnconstants.C_PVAL_SCORE], row[vnnconstants.RLIPP_SCORE]]
            for term, row in aggregated_data.iterrows()
        }

        return aggregated_dict

    def _get_scores_for_disease(self, disease):
        """
        Retrieves prediction scores for a specific disease, returning a dictionary mapping
        each term to its P_rho score for the given disease.

        :param disease: The disease or cancer type for which scores are requested.
        :type disease: str
        :return: A dictionary with Term as keys and P_rho scores as values for the specified disease.
        :rtype: dict
        """
        data = pd.read_csv(self._get_rlipp_out_dest_file(), sep='\t')
        filtered_data = data[data['Disease'] == disease]
        if filtered_data.empty:
            return {}

        scores = {
            term: [row[vnnconstants.PRHO_SCORE], row[vnnconstants.P_PVAL_SCORE],
                   row[vnnconstants.CRHO_SCORE], row[vnnconstants.C_PVAL_SCORE], row[vnnconstants.RLIPP_SCORE]]
            for term, row in filtered_data.set_index('Term').iterrows()
        }

        return scores

    def _upload_to_ndex_if_credentials_provided(self, interactome_list, root_id):
        """
        Uploads hierarchy and parent network to NDEx if credentials are provided.

        This method checks if the NDEx server, user, and password credentials are provided.
        If they are, it uploads the hierarchy and parent network to NDEx. If the parent
        network is not specified, it raises an error. If the password is specified as '-',
        it prompts the user to enter the password interactively.
        """
        if self._ndexserver and self._ndexuser and self._ndexpassword:

            if self._ndexpassword == '-':
                self._ndexpassword = getpass.getpass(prompt="Enter NDEx Password: ")

            ndex_uploader = NDExHierarchyUploader(self._ndexserver, self._ndexuser,
                                                  self._ndexpassword, self._visibility)

            if self.parent_network is None:
                logger.warning("Parent network was not specified. Hierarchy will not be in cell view.")
                cx_factory = RawCX2NetworkFactory()
                hierarchy_network = cx_factory.get_cx2network(self._get_hierarchy_dest_file())
                _, hierarchyurl = ndex_uploader._save_network(hierarchy_network)
            else:
                if interactome_list is not None:
                    for system_id, (system_interactome, _) in interactome_list.items():
                        system_interactome_uuid, _ = ndex_uploader._save_network(system_interactome)
                        self.styled_hierarchy.add_node_attribute(system_id, key='HCX::interactionNetworkUUID',
                                                                 value=system_interactome_uuid)
                        if system_id == root_id:
                            self.styled_hierarchy = ndex_uploader._update_hcx_annotations(self.styled_hierarchy,
                                                                                          system_interactome_uuid)
                    if root_id is not None:
                        _, hierarchyurl = ndex_uploader._save_network(self.styled_hierarchy)
                    else:
                        _, _, _, hierarchyurl = ndex_uploader.save_hierarchy_and_parent_network(self.styled_hierarchy,
                                                                                                self.parent_network)
                elif os.path.isfile(self.parent_network):
                    _, _, _, hierarchyurl = ndex_uploader.upload_hierarchy_and_parent_network_from_files(
                        hierarchy_path=self._get_hierarchy_dest_file(), parent_path=self.parent_network)
                else:
                    cx_factory = RawCX2NetworkFactory()
                    hierarchy_network = cx_factory.get_cx2network(self._get_hierarchy_dest_file())
                    _, _, _, hierarchyurl = ndex_uploader.save_hierarchy_and_parent_network(hierarchy_network,
                                                                                            self.parent_network)

            print(f'Hierarchy uploaded. To view hierarchy on NDEx please paste this URL in your '
                  f'browser {hierarchyurl}. To view Hierarchy on new experimental Cytoscape on the Web, go to '
                  f'{ndex_uploader.get_cytoscape_url(hierarchyurl)}')

    @staticmethod
    def _annotate_with_score(hierarchy, original_hierarchy, node_id, score_name, score):
        """
        Adds a score attribute to a node in both the hierarchy and the original hierarchy (if provided).

        :param hierarchy: The primary hierarchy network to annotate
        :param original_hierarchy: The original hierarchy network (optional)
        :param node_id: Node identifier to which the attribute is added
        :param score_name: Name of the attribute (e.g., P_rho)
        :param score: Score value to be annotated (float)
        """
        hierarchy.add_node_attribute(node_id, score_name, score, datatype='double')
        if original_hierarchy is not None:
            original_hierarchy.add_node_attribute(node_id, score_name, score, datatype='double')

    def _annotate_hierarchy(self, annotation_dict, hierarchy, original_hierarchy):
        """
        Annotates the hierarchy with P_rho scores from the given annotation dictionary,
        updating node attributes within the hierarchy file.

        :param annotation_dict: A dictionary mapping terms to their P_rho scores.
        :type annotation_dict: dict
        :param hierarchy:
        :param original_hierarchy:
        """
        for term, score in annotation_dict.items():
            node_id = term
            if not isinstance(term, int):
                node_id = hierarchy.lookup_node_id_by_name(term)
            if node_id is not None:
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.PRHO_SCORE, score[0])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.P_PVAL_SCORE, score[1])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.CRHO_SCORE, score[2])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.C_PVAL_SCORE, score[3])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.RLIPP_SCORE, score[4])
                self._annotate_with_score(hierarchy, original_hierarchy, node_id, vnnconstants.IMPORTANCE_SCORE,
                                          score[0])
        return hierarchy, original_hierarchy

    def _save_hierarchy_to_file(self, hierarchy, original_hierarchy):
        """
        Saves the annotated hierarchy and original hierarchy (if provided) to CX2 files.

        This method applies visualization styling from built-in CX2 templates to enhance
        the appearance of the saved networks. The styled hierarchy is also stored for
        further use (e.g., NDEx upload).

        :param hierarchy: Annotated hierarchy network
        :param original_hierarchy: Original version of the hierarchy (if available)
        """
        factory = RawCX2NetworkFactory()
        path_to_style_network = os.path.join(os.path.dirname(cellmaps_vnn.__file__), 'nest_style.cx2')
        style_network = factory.get_cx2network(path_to_style_network)
        vis_prop = style_network.get_visual_properties()
        hierarchy.set_visual_properties(vis_prop)
        hierarchy.write_as_raw_cx2(self._get_hierarchy_dest_file())
        if original_hierarchy is not None:
            original_hierarchy.set_visual_properties(vis_prop)
            original_hierarchy.write_as_raw_cx2(self._get_original_hierarchy_dest_file())
        self.styled_hierarchy = hierarchy

    def _annotate_interactomes_of_systems(self, parent_cx, hierarchy_cx):
        """
        Creates and annotates interactome subnetworks for each system node in the hierarchy.

        For each system, gene-level importance scores are read from each prediction directory,
        averaged, and added as node attributes in a new subnetwork extracted from the parent interactome.

        :param parent_cx: CX2 parent network (interactome) used as source for subnetworks
        :param hierarchy_cx: CX2 hierarchy network containing system nodes
        :return: A tuple (interactome_dict, root_id) where:
                 - interactome_dict maps system node IDs to (CX2Network, output file path)
                 - root_id is the ID of the root system node (if identified)
        """
        if parent_cx is None:
            return None, None
        interactome_dict = dict()
        hierarchy_net_attrs = copy.deepcopy(hierarchy_cx.get_network_attributes())
        for attr_to_remove in ['ndexSchema', 'HCX::modelFileCount', 'HCX::interactionNetworkUUID',
                               'HCX::interactionNetworkName']:
            if attr_to_remove in hierarchy_net_attrs:
                del hierarchy_net_attrs[attr_to_remove]
        root_id = None
        for system_id, node_obj in hierarchy_cx.get_nodes().items():

            new_subnet = CX2Network()
            factory = RawCX2NetworkFactory()
            interactome_style = factory.get_cx2network(
                os.path.join(os.path.dirname(cellmaps_vnn.__file__), 'interactome_style.cx2'))
            new_subnet.set_visual_properties(interactome_style.get_visual_properties())
            system_name = node_obj.get(ndexconstants.ASPECT_VALUES, {}).get(ndexconstants.NODE_NAME, system_id)
            hierarchy_net_attrs['description'] = 'RESULTS FOR SYSTEM ' + str(system_name)
            new_subnet.set_network_attributes(hierarchy_net_attrs)
            new_subnet.set_name(str(system_name) + ' assembly')

            # Aggregate gene scores from all model_prediction directories
            gene_scores_accumulator = {}
            count = {}
            for directory in self._model_predictions:
                scores_file = os.path.join(directory, str(system_id) + vnnconstants.SCORE_FILE_NAME_SUFFIX)
                if not os.path.isfile(scores_file):
                    continue
                df = pd.read_csv(scores_file, sep='\t')
                for _, row in df.iterrows():
                    gene = row['gene']
                    if gene not in gene_scores_accumulator:
                        gene_scores_accumulator[gene] = {
                            vnnconstants.MUTATION_IMPORTANCE_SCORE: 0.0,
                            vnnconstants.DELETION_IMPORTANCE_SCORE: 0.0,
                            vnnconstants.AMPLIFICATION_IMPORTANCE_SCORE: 0.0,
                            vnnconstants.GENE_IMPORTANCE_SCORE: 0.0
                        }
                        count[gene] = 0
                    gene_scores_accumulator[gene][vnnconstants.MUTATION_IMPORTANCE_SCORE] \
                        += row[vnnconstants.MUTATION_IMPORTANCE_SCORE]
                    gene_scores_accumulator[gene][vnnconstants.DELETION_IMPORTANCE_SCORE] \
                        += row[vnnconstants.DELETION_IMPORTANCE_SCORE]
                    gene_scores_accumulator[gene][vnnconstants.AMPLIFICATION_IMPORTANCE_SCORE] \
                        += row[vnnconstants.AMPLIFICATION_IMPORTANCE_SCORE]
                    gene_scores_accumulator[gene][vnnconstants.GENE_IMPORTANCE_SCORE] \
                        += row[vnnconstants.GENE_IMPORTANCE_SCORE]
                    count[gene] += 1

            # Average the scores
            for gene in gene_scores_accumulator:
                for key in gene_scores_accumulator[gene]:
                    gene_scores_accumulator[gene][key] /= count[gene]

            member_ids = []
            for gene, scores in gene_scores_accumulator.items():
                member_node_id = parent_cx.lookup_node_id_by_name(gene)
                if member_node_id is None:
                    continue
                interactome_node = copy.deepcopy(parent_cx.get_node(member_node_id))
                for key, val in scores.items():
                    interactome_node[ndexconstants.ASPECT_VALUES][key] = val
                new_subnet.add_node(node_id=member_node_id, attributes=interactome_node['v'],
                                    x=interactome_node['x'], y=interactome_node['y'])
                member_ids.append(member_node_id)

            edge_ids_to_add = set()
            for edge_id, edge_obj in parent_cx.get_edges().items():
                if edge_obj['s'] in member_ids and edge_obj['t'] in member_ids:
                    edge_ids_to_add.add(edge_id)

            for edge in edge_ids_to_add:
                interactome_edge = parent_cx.get_edge(edge)
                new_subnet.add_edge(edge_id=edge, source=interactome_edge['s'],
                                    target=interactome_edge['t'], attributes=interactome_edge['v'])

            if 'HCX::isRoot' in node_obj['v'] and node_obj['v']['HCX::isRoot'] is True:
                root_id = system_id

            subnet_file_name = os.path.join(self._outdir,
                                            str(system_name) + vnnconstants.SYSTEM_INTERACTOME_FILE_SUFFIX)
            new_subnet.write_as_raw_cx2(subnet_file_name)
            interactome_dict[system_id] = (new_subnet, subnet_file_name)

        return interactome_dict, root_id

    def _register_hierarchy(self, outdir, description, keywords, provenance_utils):
        """
        Register annotated hierarchy file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        hierarchy_out_file = self._get_original_hierarchy_dest_file()

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Annotated hierarchy file that was used to build '
                                                                    'VNN',
                     'description': description + ' Annotated hierarchy file that was used to build VNN',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_original_hierarchy(self, outdir, description, keywords, provenance_utils):
        """
        Register annotated hierarchy file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        hierarchy_out_file = self._get_hierarchy_dest_file()

        data_dict = {'name': os.path.basename(hierarchy_out_file) + ' Annotated hierarchy file',
                     'description': description + ' Annotated hierarchy file',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _copy_and_register_hierarchy_parent(self, outdir, description, keywords, provenance_utils):
        """
        Copies the parent network file to the output directory and registers it with FAIRSCAPE.

        :param outdir: Directory where outputs are stored
        :param description: Description for provenance metadata
        :param keywords: Keywords to associate with the registered dataset
        :param provenance_utils: Provenance utility for dataset registration
        :return: The dataset ID assigned to the registered parent network file
        """
        hierarchy_parent_out_file = os.path.join(outdir, 'hierarchy_parent.cx2')
        shutil.copy(self.parent_network, hierarchy_parent_out_file)

        data_dict = {'name': os.path.basename(hierarchy_parent_out_file) + ' Hierarchy parent network file',
                     'description': description + ' Hierarchy parent network file',
                     'keywords': keywords,
                     'data-format': 'CX2',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime('%m-%d-%Y')}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=hierarchy_parent_out_file,
                                                       data_dict=data_dict)
        return dataset_id

    def _register_rlipp_file(self, outdir, description, keywords, provenance_utils):
        """
        Registers the rlipp aggregated file with the FAIRSCAPE service for data provenance.

        :param outdir: The output directory where the outputs are stored.
        :param description: Description of the file for provenance registration.
        :param keywords: List of keywords associated with the file.
        :param provenance_utils: The utility class for provenance registration.

        :return: The dataset ID assigned to the registered file.
        """
        dest_path = self._get_rlipp_out_dest_file()
        description = description
        description += ' rlipp results file averaged with multiple models'
        keywords = keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(dest_path) + ' rlipp aggregated file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'txt',
                     'author': cellmaps_vnn.__name__,
                     'version': cellmaps_vnn.__version__,
                     'date-published': date.today().strftime(provenance_utils.get_default_date_format_str())}
        dataset_id = provenance_utils.register_dataset(outdir,
                                                       source_file=dest_path,
                                                       data_dict=data_dict)
        return dataset_id

    def _process_input_hierarchy_and_parent(self):
        """
        Initializes file paths for hierarchy, original hierarchy, and parent network.

        If paths are not explicitly provided, they are inferred from the first prediction RO-Crate directory.
        Raises an error if hierarchy cannot be found.
        """
        # HIERARCHY: Check first ro-crate for hierarchy if not specified by hierarchy flag
        if self.hierarchy is None:
            self.hierarchy = os.path.join(self._model_predictions[0], vnnconstants.HIERARCHY_FILENAME)
        if not os.path.exists(self.hierarchy):
            raise CellmapsvnnError("No hierarchy was specified or found in first ro-crate")
        self.hierarchy = os.path.abspath(self.hierarchy)

        # ORIGINAL HIERARCHY: Check first ro-crate for original hierarchy
        original_hierarchy_path = os.path.join(self._model_predictions[0],
                                               vnnconstants.ORIGINAL_HIERARCHY_FILENAME)
        if os.path.exists(original_hierarchy_path):
            self.original_hierarchy = os.path.abspath(original_hierarchy_path)

        # PARENT (INTERACTOME): Check first ro-crate for parent network if not specified by parent_network flag
        if self.parent_network is None:
            parent_network_path = os.path.join(self._model_predictions[0], vnnconstants.PARENT_NETWORK_NAME)
            if os.path.exists(parent_network_path):
                self.parent_network = parent_network_path
        if self.parent_network is not None and os.path.isfile(self.parent_network):
            self.parent_network = os.path.abspath(self.parent_network)

    def _get_hierarchy_cx(self):
        """
        Loads the CX2 hierarchy and, if available, the original hierarchy from file paths.

        :return: A tuple containing (hierarchy, original_hierarchy) CX2Network objects
        """
        ...
        factory = RawCX2NetworkFactory()
        hierarchy = factory.get_cx2network(self.hierarchy)
        original_hierarchy = None
        if self.original_hierarchy:
            original_hierarchy = factory.get_cx2network(self.original_hierarchy)

        return hierarchy, original_hierarchy

    def _get_parent_cx(self):
        """
        Loads the parent network (interactome) from file or NDEx server.

        :return: CX2Network object representing the parent network, or None if not found
        """
        factory = RawCX2NetworkFactory()
        if self.parent_network is None:
            return None
        elif os.path.isfile(self.parent_network):
            return factory.get_cx2network(self.parent_network)
        else:
            client = ndex2.client.Ndex2()
            client_resp = client.get_network_as_cx2_stream(self.parent_network)
            return factory.get_cx2network(json.loads(client_resp.content))

    @staticmethod
    def _annotate_hierarchy_edges(hierarchy):
        """
        Propagates node importance scores to the edges based on paths to the root node.

        Edges leading from high-importance nodes are annotated with the score if not already annotated.

        :param hierarchy: The CX2 hierarchy network to annotate
        :return: The modified hierarchy with updated edge attributes
        """
        node_id_score_map = {}
        for node_id, node_obj in hierarchy.get_nodes().items():
            score = node_obj[ndexconstants.ASPECT_VALUES].get(vnnconstants.IMPORTANCE_SCORE, 0)
            node_id_score_map[node_id] = score

        edge_target_map = {}

        for edge_id, edge_obj in hierarchy.get_edges().items():
            edge_target_map[edge_obj[ndexconstants.EDGE_TARGET]] = edge_obj
            hierarchy.add_edge_attribute(edge_id, key=vnnconstants.EDGE_IMPORTANCE_SCORE,
                                         value=0.0,
                                         datatype=ndexconstants.DOUBLE_DATATYPE)

        for node_id, score in node_id_score_map.items():
            if score < 0.7:
                continue
            is_root = False
            cur_node_id = node_id
            while is_root is False:
                if cur_node_id not in edge_target_map:
                    break
                edge_obj = edge_target_map[cur_node_id]
                if not edge_obj[ndexconstants.ASPECT_VALUES][vnnconstants.EDGE_IMPORTANCE_SCORE] > 0.0:
                    edge_obj[ndexconstants.ASPECT_VALUES][vnnconstants.EDGE_IMPORTANCE_SCORE] = score
                cur_node_id = edge_obj[ndexconstants.EDGE_SOURCE]
                parent_node = hierarchy.get_node(cur_node_id)
                is_root = parent_node[ndexconstants.ASPECT_VALUES].get('HCX::isRoot', False)
        return hierarchy

    def _process_rocrates(self):
        for idx, directory in enumerate(self._model_predictions):
            if not os.path.exists(os.path.join(directory, vnnconstants.RLIPP_OUTPUT_FILE)):
                self._model_predictions[idx] = os.path.join(directory, 'out_predict')

    def _add_provenance_info(self, hierarchy, provenance_utils=ProvenanceUtil(), author='cellmaps_vnn',
                             version=cellmaps_vnn.__version__):
        hierarchy.add_network_attribute(key='prov:wasGeneratedBy', value=author + ' ' + version)
        rocrate_id = provenance_utils.get_id_of_rocrate(os.path.dirname(self.hierarchy))
        if rocrate_id is not None:
            hierarchy.add_network_attribute(key='prov:wasDerivedFrom',  value='RO-crate: ' + str(rocrate_id))
        return  hierarchy
