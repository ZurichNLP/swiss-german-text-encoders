
# Data for continued pre-training

## swisscrawl
Data can be requested via https://icosys.ch/swisscrawl (not managed by UZH).

## gsw_tweets
Data are not public.

## swissdox
Data can be accessed with a license (see https://t.uzh.ch/1hI for more information).

# Data for evaluation

## GSW_test_set (POS) 
- `wget https://noe-eva.github.io/publication/acl22/GSW_test_set.zip`
- `unzip GSW_test_set.zip && rm GSW_test_set.zip`

## UD_German-HDT (POS)
- `git clone --depth 1 --branch r2.13 https://github.com/UniversalDependencies/UD_German-HDT.git`
- `cat UD_German-HDT/de_hdt-ud-train-a-1.conllu UD_German-HDT/de_hdt-ud-train-a-2.conllu > UD_German-HDT/de_hdt-ud-train-a.conllu`

## gdi-vardial-2019
Data are not public.

## NTREX (retrieval)
- `git clone --depth 1  https://github.com/MicrosoftTranslator/NTREX.git`

## dialect_eval (retrieval)
- `git clone --depth 1 https://github.com/textshuttle/dialect_eval.git`
- `cp dialect_eval/evaluation/ntrex-128/references/en-gsw_be.refA.txt NTREX/NTREX-128/newstest2019-ref.gsw-BE.txt`
- `cp dialect_eval/evaluation/ntrex-128/references/en-gsw_zh.refA.txt NTREX/NTREX-128/newstest2019-ref.gsw-ZH.txt`
