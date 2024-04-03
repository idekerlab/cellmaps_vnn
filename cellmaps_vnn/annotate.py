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

    def _get_rlipp_out_dest_file(self):
        return os.path.join(self._theargs.outdir, vnnconstants.RLIPP_OUTPUT_FILE)

    def _get_hierarchy_dest_file(self):
        return os.path.join(self._theargs.outdir, 'hierarchy.cx2')

    def _aggregate_prediction_scores_from_models(self):

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
        filepath = self._get_rlipp_out_dest_file()
        data = pd.read_csv(filepath, sep='\t')
        average_p_rho = data.groupby('Term')[vnnconstants.PRHO_SCORE].mean()
        average_p_rho_dict = average_p_rho.to_dict()

        return average_p_rho_dict

    def annotate(self, annotation_dict):
        factory = RawCX2NetworkFactory()
        hierarchy = factory.get_cx2network(self.hierarchy)
        for term, p_rho in annotation_dict.items():
            node_id = hierarchy.lookup_node_id_by_name(term)
            if node_id is not None:
                hierarchy.add_node_attribute(node_id, vnnconstants.PRHO_SCORE, p_rho, datatype='double')
        hierarchy.write_as_raw_cx2(self._get_hierarchy_dest_file())

    def run(self):
        self._aggregate_prediction_scores_from_models()
        annotation_dict = self._aggregate_scores_from_diseases()
        self.annotate(annotation_dict)

    def register_outputs(self, outdir, description, keywords, provenance_utils):
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


