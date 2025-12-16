# OSD2E-SIM-NAS
1. **The `Diffevo for GAN` folder** (for GAN architecture search tasks)  
   - `diffevo-search_gen_arch.py`: code for the search phase  
   - `diffevo-fully_train_arch.py`: code for the re-training phase  
   - Files prefixed with `STL10` are for the STL-10 dataset  

2. **The `requirements.txt` file**  
   - Lists the required libraries and their corresponding versions  

3. **The `onestep-DiffEvo-NAS` folder** (for image classification tasks)  
   - The three files starting with `onestep` are the search-phase codes for:  
     - CIFAR-10  
     - CIFAR-100  
     - NAS-Bench-201  
   - `train_search.py`: re-training code
