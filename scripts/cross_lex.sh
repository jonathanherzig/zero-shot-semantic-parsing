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

version='cross_lex'
domains='blocks calendar socialnetwork publications restaurants housing recipes'

for domain in $domains
do
  python src/py/main.py -d 200 -i 100 -o 100 -p attention -u 1 -t 15,5,5,5 -c lstm -m attention --stats-file res/stats_${version}_${domain}.json --domain overnight-${domain} -k 5 --dev-seed 0 --model-seed 0 --train-data ${data_path}/cross_domain/${domain}_train.tsv --dev-data ${data_path}/cross_domain/${domain}_test.tsv --save-file params/params_${version}_${domain}
done
