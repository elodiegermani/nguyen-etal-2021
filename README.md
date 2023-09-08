# nguyen-etal-2021
Notebook for the reproduction of: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8127393/#SD1. 


## Dockerfile 

```bash
neurodocker generate docker \
--pkg-manager yum \
--base-image fedora:36 \
--install git \
--ants version=2.3.4 method=binaries \
--fsl version=6.0.6.1 method=binaries \
--afni method=binaries version=latest install_r_pkgs=true \
--workdir /home \
--miniconda version=latest \
        env_name=neuro \
        env_exists=false \
        conda_install="python=3.10 traits jupyter nilearn graphviz nipype scikit-image seaborn" \
        pip_install="matplotlib" \
--env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
--run-bash "source activate neuro" \
--user=neuro \
--user=root \
--run 'chmod 777 -Rf /home' \
--run 'chown -R neuro /home' \
--run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile 

```