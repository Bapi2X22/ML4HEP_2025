{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9a416a67",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import h5py\n",
    "import matplotlib.pyplot as plt\n",
    "import awkward as ak\n",
    "import os\n",
    "from collections import defaultdict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4b4cfc0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# train_dataset = h5py.File(\"hgcal_electron_data_0001.h5\", \"r\")\n",
    "# # nhits = np.array(train_dataset['nhits'])\n",
    "# # target_energy = np.array(train_dataset['target'])\n",
    "# Rechit_z = np.array(train_dataset['rechit_z'])\n",
    "# Rechit_Energy = np.array(train_dataset['rechit_energy'])\n",
    "# Rechit_x = np.array(train_dataset['rechit_x'])\n",
    "# Rechit_y = np.array(train_dataset['rechit_y'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3dee7770",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the HDF5 file\n",
    "with h5py.File(\"hgcal_electron_data_0001.h5\", \"r\") as f:\n",
    "    nhits = f[\"nhits\"][:]               # shape: (num_events,)\n",
    "    x = f[\"rechit_x\"][:]                # shape: (total_hits,)\n",
    "    y = f[\"rechit_y\"][:]\n",
    "    z = f[\"rechit_z\"][:]\n",
    "    E = f[\"rechit_energy\"][:]\n",
    "\n",
    "# Convert nhits to integer if needed\n",
    "nhits = nhits.astype(np.int64)\n",
    "\n",
    "# Use ak.unflatten to split each feature eventwise\n",
    "events = ak.Array({\n",
    "    \"x\": ak.unflatten(x, nhits),\n",
    "    \"y\": ak.unflatten(y, nhits),\n",
    "    \"z\": ak.unflatten(z, nhits),\n",
    "    \"E\": ak.unflatten(E, nhits),\n",
    "})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f6c6c03d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1.9435272216796875"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "events[0]['x'][0]"
   ]
  }
 ],
 "metadata": {
  "@webio": {
   "lastCommId": null,
   "lastKernelId": null
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
