import logging
import os
from datetime import date

import numpy as np
import pandas as pd
from cellmaps_utils import constants

import cellmaps_vnn
import cellmaps_vnn.constants as vnnconstants
from ndex2.cx2 import RawCX2NetworkFactory

from cellmaps_vnn.exceptions import CellmapsvnnError

logger = logging.getLogger(__name__)


class VNNAnnotate:
    COMMAND = 'annotate'

    def __init__(self, theargs):
        """
        Constructor. Sets up the hierarchy path either directly from the arguments or by looking for
        a hierarchy.cx2 file in the first RO-Crate directory provided. If neither is found, raises an error.

        :param theargs: The arguments provided to the command line interface.
        :type theargs: argparse.Namespace
        :raises CellmapsvnnError: If no hierarchy path is specified or found.
        """
        self._theargs = theargs
        if theargs.hierarchy is not None:
            self.hierarchy = theargs.hierarchy
        else:
            hierarchy_path = os.path.join(theargs.model_predictions[0], 'hierarchy.cx2')
            if os.path.exists(hierarchy_path):
                self.hierarchy = hierarchy_path
            else:
                raise CellmapsvnnError("No hierarchy was specified or found in first ro-crate")

    @staticmethod
    def add_subparser(subparsers):
        """
        Adds a subparser for the 'annotate' command.
        """
        # TODO: modify description later
        desc = """
        Version: todo

        The 'annotate' command takes ..
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
        parser.add_argument('--hierarchy', help='Path to hierarchy (optional), if not set the hierarchy will be '
                                                'selected from the first RO-Crate passed in --model_predictions '
                                                'argument', type=str)
        parser.add_argument('--disease', help='Specify the disease or cancer type for which the annotations will be '
                                              'focused. This allows the annotation process to tailor the results '
                                              'according to the particular disease or cancer type. If not set, '
                                              'prediction scores for all diseases will be aggregated .', type=str)
        # optional: --upload with --parent_network required if --upload set

    def _get_rlipp_out_dest_file(self):
        """
        Constructs the file path for the RLIPP output file within the specified output directory.

        :return: The file path for the RLIPP output file.
        :rtype: str
        """
        return os.path.join(self._theargs.outdir, vnnconstants.RLIPP_OUTPUT_FILE)

    def _get_hierarchy_dest_file(self):
        """
        Constructs the file path for the hierarchy output file within the specified output directory.

        :return: The file path for the hierarchy output file.
        :rtype: str
        """
        return os.path.join(self._theargs.outdir, 'hierarchy.cx2')

    def _aggregate_prediction_scores_from_models(self):
        """
        Aggregates prediction scores from multiple models' outputs by averaging them.
        The aggregated scores are then saved to the RLIPP output destination file.
        """
        data = {}

        for directory in self._theargs.model_predictions:
            filepath = os.path.join(directory, vnnconstants.RLIPP_OUTPUT_FILE)
            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith('Term') or not line.strip():
                        continue

                    parts = line.strip().split('\t')
                    key = (parts[0], parts[-1])  # (Term, Disease)
                    values = np.array([float(v) for v in parts[1:-1]])

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
        filepath = self._get_rlipp_out_dest_file()
        data = pd.read_csv(filepath, sep='\t')
        average_p_rho = data.groupby('Term')[vnnconstants.PRHO_SCORE].mean()
        average_p_rho_dict = average_p_rho.to_dict()

        return average_p_rho_dict

    def _get_scores_for_disease(self, disease):
        """
        Retrieves prediction scores for a specific disease, returning a dictionary mapping
        each term to its P_rho score for the given disease.

        :param disease: The disease or cancer type for which scores are requested.
        :type disease: str
        :return: A dictionary with Term as keys and P_rho scores as values for the specified disease.
        :rtype: dict
        """
        filepath = self._get_rlipp_out_dest_file()
        data = pd.read_csv(filepath, sep='\t')
        filtered_data = data[data['Disease'] == disease]
        if filtered_data.empty:
            return {}
        scores = filtered_data.set_index('Term')[vnnconstants.PRHO_SCORE].to_dict()

        return scores

    def annotate(self, annotation_dict):
        """
        Annotates the hierarchy with P_rho scores from the given annotation dictionary,
        updating node attributes within the hierarchy file.

        :param annotation_dict: A dictionary mapping terms to their P_rho scores.
        :type annotation_dict: dict
        """
        factory = RawCX2NetworkFactory()
        hierarchy = factory.get_cx2network(self.hierarchy)
        for term, p_rho in annotation_dict.items():
            node_id = hierarchy.lookup_node_id_by_name(term)
            if node_id is not None:
                hierarchy.add_node_attribute(node_id, vnnconstants.PRHO_SCORE, p_rho, datatype='double')

        # TODO: apply styles to hierarchy
        # style_cx2 = factory.get_cx2network(os.path.join(os.path.dirname(cellmaps_vnn.__file__),
        #                                                 'NeST-VNN_Palbociclib.cx2'))
        # vis_prop = style_cx2.get_visual_properties()
        # hierarchy.set_visual_properties(vis_prop)

        hierarchy.write_as_raw_cx2(self._get_hierarchy_dest_file())

    def run(self):
        """
        The logic for annotating hierarchy with prediction results from cellmaps_vnn. It aggregates prediction scores
        from models, optionally filters them for a specific disease, and annotates the hierarchy with these scores.
        """
        self._aggregate_prediction_scores_from_models()
        if self._theargs.disease is None:
            annotation_dict = self._aggregate_scores_from_diseases()
        else:
            annotation_dict = self._get_scores_for_disease(self._theargs.disease)
        self.annotate(annotation_dict)

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
        return [hierarchy_id, rlipp_id]

    def _register_hierarchy(self, outdir, description, keywords, provenance_utils):
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

