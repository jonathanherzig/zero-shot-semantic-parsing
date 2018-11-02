if [ $1 = 'dev' ]
then
    data_path='overnight_delex_dev'
    split=--split_dev
else
if [ $1 = 'test' ]
then
    data_path='overnight_delex'
    split=''
else
    echo 'arg0 should be dev or test'
    exit 0
fi
fi

version='in_abstract'
domains='blocks calendar socialnetwork publications restaurants housing recipes'

for domain in $domains
do
  python src/py/zero_shot/aligner.py -d $domain --eval_dev -t 30 -c 200 --output_keep_prob 1.0 --input_keep_prob 0.8
  python src/py/main.py -d 250 -i 100 -o 100 -p attention -u 1 -t 15,5,5,5 -c lstm -m attention --stats-file res/stats_${version}_${domain}.json --domain overnight-${domain} -k 5 --dev-seed 0 --model-seed 0 --train-data ${data_path}/${domain}_train.tsv --dev-data ${data_path}/${domain}_test.tsv --save-file params/params_${version}_${domain} --delexicalize -l 0.001
done
