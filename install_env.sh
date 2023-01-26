module load Python/3.8.6-GCCcore-10.2.0

python3 -m venv venv

source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
git clone --recurse-submodules git@github.com:hahnec/multimodal_emg_private multimodal_emg_repo
git clone --recurse-submodules git@github.com:hahnec/simple_tracker simple_tracker_repo
python3 -m pip install -r multimodal_emg_repo/requirements.txt

ln -sf ./multimodal_emg_repo/multimodal_emg multimodal_emg
ln -sf ./simple_tracker_repo/simple_tracker simple_tracker
