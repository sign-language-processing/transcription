import csv
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, default='./pose_to_segments/src/summary_pro.csv')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--preliminary', action='store_true')
args = parser.parse_args()

# LaTeX table structure
if args.verbose:
    if args.preliminary:
        latex_table=r"""
\begin{tabular}{lllccc|ccc}
\toprule
& & & \multicolumn{3}{c}{\textbf{Sign}} & \multicolumn{3}{c}{\textbf{Phrase}}\\
\cmidrule(lr){4-6} \cmidrule(lr){7-9}
\multicolumn{3}{l}{\textbf{Experiment}} & \textbf{F1} & \textbf{IoU} & \textbf{\%} & \textbf{F1} & \textbf{IoU} & \textbf{\%}\\
\midrule
"""
    else:
        latex_table = r"""
\begin{table*}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllccc|ccc|cc}
\toprule
& & & \multicolumn{3}{c}{\textbf{Sign}} & \multicolumn{3}{c}{\textbf{Phrase}} & \multicolumn{2}{c}{\textbf{Efficiency}} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-11}
\multicolumn{3}{l}{\textbf{Experiment}} & \textbf{F1} & \textbf{IoU} & \textbf{\%} & \textbf{F1} & \textbf{IoU} & \textbf{\%} & \textbf{\#Params} & \textbf{Time} \\
\midrule
"""
else:
    latex_table = r"""
\begin{table*}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccc|ccc|cc}
\toprule
& & \multicolumn{3}{c}{\textbf{Sign}} & \multicolumn{3}{c}{\textbf{Phrase}} & \multicolumn{2}{c}{\textbf{Efficiency}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}
\multicolumn{2}{l}{\textbf{Experiment}} & \textbf{F1} & \textbf{IoU} & \textbf{\%} & \textbf{F1} & \textbf{IoU} & \textbf{\%} & \textbf{\#Params} & \textbf{Time} \\
\midrule
"""

with open(args.input, 'r') as f:
    reader = csv.DictReader(f)  # use DictReader
    for row_dict in reader:
        if row_dict['id'] in ['E1', 'E1s', 'E4ba']:
            latex_table += "\\midrule\n"

        if args.preliminary and row_dict['id'] in ['E2', 'E3', 'E4', 'E5']:
            latex_table += "\\midrule\n"

        # Remove std
        if not args.verbose:
            for k, v in row_dict.items():
                row_dict[k] = v.split('±')[0]

        if args.preliminary:
            row_dict['id'] = row_dict['id'].replace('E', 'P')
            row_dict['note'] = row_dict['note'].replace('E', 'P')
        
        row_dict['test_sign_frame_f1'] = f"${row_dict['test_sign_frame_f1']}$"
        row_dict['test_sentence_frame_f1'] = f"${row_dict['test_sentence_frame_f1']}$"
        row_dict['test_sign_segment_IoU'] = f"${row_dict['test_sign_segment_IoU']}$"
        row_dict['test_sign_segment_percentage'] = f"${row_dict['test_sign_segment_percentage']}$"
        row_dict['test_sentence_segment_percentage'] = f"${row_dict['test_sentence_segment_percentage']}$"
        row_dict['test_sentence_segment_IoU'] = f"${row_dict['test_sentence_segment_IoU']}$"
        
        if row_dict['id'] in ['E0', 'P0', 'P0.1']:
            row_dict['test_sign_frame_f1'] = '---'
            row_dict['test_sentence_frame_f1'] = '---'
        
        latex_table += r"\textbf{" + row_dict['id'] + r"} & \textbf{" + row_dict['note'].replace('_', '\\_') + "}" + " & " 
        if args.verbose:
            latex_table += r"\textbf{test} &"
        latex_table += row_dict['test_sign_frame_f1'] + " & " + row_dict['test_sign_segment_IoU'] + " & " + row_dict['test_sign_segment_percentage'] + " & "
        latex_table += row_dict['test_sentence_frame_f1'] + " & " + row_dict['test_sentence_segment_IoU'] + " & " + row_dict['test_sentence_segment_percentage']
        if not args.preliminary:
            latex_table += " & " + row_dict['#parameters'] + " & " + row_dict['training_time_avg']
        latex_table += r"\\" + "\n"

        if args.verbose:
            row_dict['dev_sign_frame_f1'] = f"${row_dict['dev_sign_frame_f1']}$"
            row_dict['dev_sentence_frame_f1'] = f"${row_dict['dev_sentence_frame_f1']}$"
            row_dict['dev_sign_segment_IoU'] = f"${row_dict['dev_sign_segment_IoU']}$"
            row_dict['dev_sign_segment_percentage'] = f"${row_dict['dev_sign_segment_percentage']}$"
            row_dict['dev_sentence_segment_percentage'] = f"${row_dict['dev_sentence_segment_percentage']}$"
            row_dict['dev_sentence_segment_IoU'] = f"${row_dict['dev_sentence_segment_IoU']}$"

            if row_dict['id'] in ['E0', 'P0', 'P0.1']:
                row_dict['dev_sign_frame_f1'] = '---'
                row_dict['dev_sentence_frame_f1'] = '---'

            if row_dict['id'] == 'E0':
                row_dict['dev_sign_frame_f1'] = '---'
                row_dict['dev_sentence_frame_f1'] = '---'

            latex_table += r"& & \textbf{dev} &"
            latex_table += row_dict['dev_sign_frame_f1'] + " & " + row_dict['dev_sign_segment_IoU'] + " & " + row_dict['dev_sign_segment_percentage'] + " & "
            latex_table += row_dict['dev_sentence_frame_f1'] + " & " + row_dict['dev_sentence_segment_IoU'] + " & " + row_dict['dev_sentence_segment_percentage']
            if not args.preliminary:
                latex_table += " & " + row_dict['#parameters'] + " & " + row_dict['training_time_avg']
            latex_table += r"\\" + "\n"


latex_table += r"""
\bottomrule
\end{tabular}
}
\caption{Mean test evaluation metrics for our experiments. The best score of each column is in bold. Appendix \ref{appendix:preliminary} contains a complete report including validation metrics and standard deviation of all experiments.}
\label{tab:results}
\end{table*}
"""

# Write the LaTeX table to a .tex file
print(latex_table.replace('±', '\pm'))
