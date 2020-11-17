Using ``phys2bids`` for massive Amount of data
############################################
We have seen that ``phys2bids`` could automatically cut  parts of the physiological signals that we are interested in (i.e. where the trigger pulse indicates neuroimaging acquisitions). We have also seen that this workflow could be applied to "multi-run" neuroimaging acquisition sessions.

 Now, let us outline how to apply the workflow to multiple physiological acquisition files procedurally.

 The goal of such procedure is to reduce the time-consuming steps required to match (f)MRI acquisitions with their proper physiological acquisition.

.. note::
      This is time consuming because, as we have seen, ``phys2bids`` has to know the number of trigger timepoints in order to deal with multi-run acquisitions. Of course, we do not want to manually input this parameter for each session.

.. warning::
    We still have to implement smarter solutions that require less patch working (maybe physiopy will have a parent workflow that calls all its tools, making a full physiological data preparation pipeline; help us !).


``phys2bids``'s Command Line Interface parameters
----------------------------------------------------
* `-in` : *name of input file* - easy to find, but still have to get a list
* `-chtrig` : *# of trigger channel* - if recording device is stable, no change
* `-ntp` : *# of trigger timepoints per run* - tricky; we have to look for it in the metadata (`_bold.json`, `_<sequence>.json` of each neuroimaging acquisition run)
* `-tr` : *length of Repetition Time in seconds* - preferably, we want to batch process sessions that contained the same MRI sequence
* `-outdir` : *path inside BIDS dataset* - BIDS is still liberal to that matter. Do you want your BIDS-compatible raw segments (taskNN_run01.tsv.gz) to live under sourcedata or directly beside your BIDS compatible raw MRI data - or clean MRI data ?
* `-heur` :

Fetching the information we need from BIDS dataset
==================================================
The information we need to fetch is essentially the *number of volumes* in each run, and for that : we need to link neuroimaging and physiological acquisitions

``phys2bids`` is meant to be used in conjunction with other of the BIDS ecosystem. That is, ``phys2bids`` workflow relies on the metadata given by other BIDS processes. For instance, researchers may use `dicom2bids <http://nipy.org/workshops/2017-03-boston/lectures/bids-heudiconv/#1>`_ to convert their neuroimaging data to BIDS format.

Then, researchers can use datetime information to link the appropriate physiological acquisition files with a specific neuroimaging acquisition. There are actually a number of ways to work this out, and it depends on the researcher's storage organization.

 (NOTE : *maybe link [match_acq_bids.py] on ds_prep neuromod*)

The File structure needed
-------------------------
At this point, what is imperative for ``phy2bids`` users to have in order to process files in batch is a BIDS root containing *n subject* directories (with BIDS compatible neuroimaging files), and also, a *sourcedata* directory.

::

    BIDS-dataset
    ├── CHANGES
    ├── dataset_description.json
    ├── derivatives
    ├── participants.tsv
    ├── README
    ├── sourcedata
    │   └── physio
    |   |   ├── sub-01
    |   |   |   ├──ses-001
    |   |   │   ├── sub-01_ses-001_physio_fmri_matches.tsv
    |   |   │   └── sub-01_ses-bournesup01.acq
    |   |   ├── sub-nn
    ├── stimuli
    │   ├── task
    ├── sub-01
    |   ├── ses-001
    |   |   ├── anat
    |   |   ├── fmap
    |   │   ├── func
    |   |   |   ├── sub-01_ses-001_task-movie01_bold.json
    |   |   |   ├── sub-01_ses-001_task-movie01_bold.nii.gz
    |   |   |   ├── sub-01_ses-001_task-movie02_bold.json
    |   │   |   └── sub-01_ses-001_scans.tsv ->
    |   ├── ses-nnn
    ├── sub-nn

``root/sub-01/`` contains all BIDS compatible outputs for neuroimaging data. This is where we will find the information we need for each session.

``sourcedata`` is where raw/uncut physiological acquisition files can live. They are supposed to follow another *n subject* hierarchy that will also contain ``_physio_fmri_matches.tsv`` in each session. These matches table files point to the proper neuroimaging acquisitions.

**Example** :

+------------------------------------+------------------------------------+
| (f)MRI acquisitions                | physiological acquisitions         |
+====================================+====================================+
| path/to/func/                      | path/to/                           |
| sub-01_ses-008_taskNN_run01.nii.gz | generic_name_datetime.acq          |
|------------------------------------+------------------------------------|
| path/to/func/                      | path/to/                           |
| sub-01_ses-008_taskNN_run02.nii.gz | generic_name_datetime.acq          |
|------------------------------------+------------------------------------|
| ...                                | ...                                |
+------------------------------------+------------------------------------+


Getting number of volumes per run
---------------------------------
We can use the path given here (in ``_physio_fmri_matches.tsv``) to access metadata of each run, in each session, for each subject.

* List a specific file type in a subject directory by sessions.
    * ``courtois-neuromod/ds_prep/physio/code/utils/list_sub.py``
    * you can list both physiological acquisition files and the matches tables, using this utility code.

* Get trigger information for each runs
    * ``courtois-neuromod/ds_prep/physio/code/utils/get_info.py``
    * you can fetch information from the json files in each session using this utility code.
