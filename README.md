# OSD2E-SIM-NAS
**Instruction**
1. **The `Diffevo for GAN` folder** (for GAN architecture search tasks)  
   - `diffevo-search_gen_arch.py`: code for the search phase  
   - `diffevo-fully_train_arch.py`: code for the retraining phase  
   - Files prefixed with `STL10` are for the STL-10 dataset  
   - `requirements.txt`: lists the required libraries and their corresponding versions for this task  

2. **The `onestep-DiffEvo-NAS` folder** (for image classification tasks)  
   - The three files starting with `onestep` are the search-phase codes for:  
     - CIFAR-10  
     - CIFAR-100  
     - NAS-Bench-201  
   - `train_search.py`: retraining code  

3. **The `diffevo` folder** (core diffusion evolutionary algorithm implementation)  
   - Contains the foundational implementation of the diffusion evolutionary algorithm, including files such as:  
     - `generator.py`: generator-related code  
     - `optimizer.py`: optimizer-related code
