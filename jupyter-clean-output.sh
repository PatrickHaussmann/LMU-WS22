if [[ $# -ne 1 ]] || [[ $1 != *.ipynb ]]; then
    echo "One .ipynb file as argument is needed"
    exit 1
fi

cat <<< $(jq --indent 1 ' (.cells[] | select(has("outputs")) | .outputs) = [] | (.cells[] | select(has("execution_count")) | .execution_count) = null | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}} | .cells[].metadata = { } ' $1) > $1
