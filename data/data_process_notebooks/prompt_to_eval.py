import argparse
import re

import pandas as pd


def get_scores(eval_result: str) -> list:
    """Get evaluation scores from GPT-4 evaluation output."""
    score_list = re.findall(r': (\d+\.?\d*)', eval_result)
    return list(map(float, score_list))


def parse_output(
    input_path: str = 'eval_gpt_3.5_multi_prompt_score.csv',
    output_path: str = 'eval_gpt_3.5_multi_result.csv',
) -> None:
    """Parse GPT-4 evaluation output."""
    df = pd.read_csv(input_path)

    column_names = [
        'understandable',
        'natural',
        'maintains_context',
        'interesting',
        'uses_knowledge',
        'empathy',
        'conversational',
        'overall_quality'
    ]
    df[column_names] = pd.DataFrame(df['score'].apply(get_scores).to_list(), index=df.index)

    avg_scores = df[column_names].mean()
    new_row = pd.Series(["평균", "평균"] + list(avg_scores), index=df.columns)
    df.loc[len(df)] = new_row

    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='GPT-4 Evaluation Output Parser')
    parser.add_argument('--input_path', type=str, default='eval_gpt_3.5_multi_prompt_score.csv',
                        help='File path to parse GPT-4 evaluation output')
    parser.add_argument('--output_path', type=str, default='eval_gpt_3.5_multi_result.csv',
                        help='Output path of parsed GPT-4 evaluation scores')

    args = parser.parse_args()

    parse_output(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
