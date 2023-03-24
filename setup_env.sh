conda install --yes --file requirements.txt

conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install pytorch3d -c pytorch3d

pip install -e .

cd ..
