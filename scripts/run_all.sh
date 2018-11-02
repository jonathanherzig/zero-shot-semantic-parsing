split=dev

sh scripts/in_abstract.sh $split
sh scripts/zero_shot.sh $split
sh scripts/in_lex.sh $split
sh scripts/cross_lex.sh $split
sh scripts/cross_lex_rep.sh $split