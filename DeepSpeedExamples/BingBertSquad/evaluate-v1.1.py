import argparse
import json
import evaluate as eval

if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(description='Evaluation for SQuAD ' +
                                     expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()

    print(
        json.dumps(
            eval.evaluate(expected_version, args.dataset_file,
                          args.prediction_file)))
