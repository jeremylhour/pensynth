pip install -r requirements.txt

echo DOWNLOAD LALONDE 1986 DATASET
cd ../data
Rscript downloadLalondeData.R

cd ../incremental_algo_puresynth/
python3 main.py