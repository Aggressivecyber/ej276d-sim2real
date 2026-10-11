#!/usr/bin/env bash
set -eu
python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
"${python}" -B ./pipeline_receipt.py started
for entry in generate_full_response.py replay_full_response.py evaluate_saved.py summarize_and_plot.py
do
    "${python}" -B ./pipeline_receipt.py progress "${entry}"
    if "${python}" -B ./linux_supervise.py "${entry}"
    then
        :
    else
        code=$?
        "${python}" -B ./pipeline_receipt.py failed "${entry}" "${code}"
        exit "${code}"
    fi
done
"${python}" -B ./pipeline_receipt.py completed
