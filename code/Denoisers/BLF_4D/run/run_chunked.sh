#!/usr/bin/env bash

generalError()
{
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


# test input
if [[ $# -ne 1 ]]; then
	>&2 echo "ERROR: Input master file!"
	exit 1
fi


denoisingMasterFile=$(readlink -f ${1})
if [[ ! -e ${denoisingMasterFile} ]]; then echo "denoisingMasterFile does not exist!"; exit 1; fi


cd $(dirname ${0})

chunkedDenoise_script=$(readlink -f "../../../ChunkedDenoise/ChunkedDenoise_4D/chunkedDenoise_4D.sh")
chunkedDenoiseMasterFile=$(readlink -f "../../../../control/ChunkedDenoise/ChunkedDenoise_4D/master.txt")

plainParamsFile=$(readlink -f ../../../plainParams/plainParams.sh)
copy_fileList_file=$(readlink -f ../../../plainParams/Cloning/copy_fileList.sh)

denoiserCloneFile=$(readlink -f "../../ModularCode_4D/clone_denoiser_4D.sh")
denoiseScript=$(readlink -f "../C_Code/Recon/runRecon.sh")
denoisingMultiRun=$(readlink -f "../../multiRun.sh")
prefix="chunked_"
suffix=""



if [[ ! -e ${chunkedDenoise_script} ]]; then messageError "chunkedDenoise_script does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${chunkedDenoiseMasterFile} ]]; then messageError "chunkedDenoiseMasterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoisingMasterFile} ]]; then messageError "denoisingMasterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoiserCloneFile} ]]; then messageError "denoiserCloneFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoiseScript} ]]; then messageError "denoiseScript does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoisingMultiRun} ]]; then messageError "denoisingMultiRun does not exist!"; generalError "$0 $@"; exit 1; fi

# get denoiser file lists paths

noisyBinaryFName_timeList=$(bash ${plainParamsFile} -a get -m ${denoisingMasterFile} -F "noisyBinaryFName_timeList" -r)
if [[  $? != 0 ]]; then messageError "getting noisyBinaryFName_timeList" ; generalError "$0 $@"; exit 1; fi

denoisedBinaryFName_timeList=$(bash ${plainParamsFile} -a get -m ${denoisingMasterFile} -F "denoisedBinaryFName_timeList" -r)
if [[  $? != 0 ]]; then messageError "getting denoisedBinaryFName_timeList" ; generalError "$0 $@"; exit 1; fi

# get denoiser file lists paths

chunkedDenoiser_noisyBinaryFName_timeList=$(bash ${plainParamsFile} -a "get" -m ${chunkedDenoiseMasterFile} -F "noisyBinaryFName_timeList" -r)
if [[  $? != 0 ]]; then messageError "getting noisyBinaryFName_timeList" ; generalError "$0 $@"; exit 1; fi

chunkedDenoiser_denoisedBinaryFName_timeList=$(bash ${plainParamsFile} -a "get" -m ${chunkedDenoiseMasterFile} -F "denoisedBinaryFName_timeList" -r)
if [[  $? != 0 ]]; then messageError "getting denoisedBinaryFName_timeList" ; generalError "$0 $@"; exit 1; fi


# Copy filelists

bash ${copy_fileList_file} -s $noisyBinaryFName_timeList -d $chunkedDenoiser_noisyBinaryFName_timeList
bash ${copy_fileList_file} -s $denoisedBinaryFName_timeList -d $chunkedDenoiser_denoisedBinaryFName_timeList


time bash "${chunkedDenoise_script}" -M "${chunkedDenoiseMasterFile}" -m "${denoisingMasterFile}" -c "${denoiserCloneFile}" -d "${denoiseScript}" -r "${denoisingMultiRun}" -p "${prefix}" -s "${suffix}" -L "shell"

