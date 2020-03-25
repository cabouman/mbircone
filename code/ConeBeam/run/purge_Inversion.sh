#!/usr/bin/env bash

generalError()
{
    >&2 echo "purge_Inversion error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


while getopts ":M:m:" option
do
    case $option in
        M) master=$(readlink -f "${OPTARG}");;
        m) mode="${OPTARG}";;
        ?)
            >&2 echo "    Unknown option -${OPTARG}!"           
            exit 1;;
    esac
done

if [[ ! ${mode} == "deep" ]] && [[ ! ${mode} == "shallow" ]]; then
    >&2 echo "Unknown mode \"${mode}\""
    generalError "$0 $@"; exit 1
fi

executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[ ! -e ${plainParamsFile} ]]; then >&2 echo "plainParamsFile ${plainParamsFile} does not exist!"; generalError "$0 $@" ; exit 1; fi




if [[ ${mode} == "deep" ]]; then
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f sino -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f driftSino -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f origSino -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f wght -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f errsino -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f sinoMask -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f recon -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f reconROI -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
        
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f proxMapInput -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f lastChange -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f timeToChange -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f phantom -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"
    
    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f sysMatrix -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"    

    tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -f wghtRecon -r)
    [[ -e ${tempFName} ]] && rm "${tempFName}"    
fi


tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F dataSetInfo -r)
rm "${tempFName}"

tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F preprocessingParams -r)
rm "${tempFName}"

tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F binaryFNames -r)
rm "${tempFName}"

tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F sinoParams -r)
rm "${tempFName}"

tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F imgParams -r)
rm "${tempFName}"

tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F reconParams -r)
rm "${tempFName}"

tempFName=$(bash "${plainParamsFile}" -a get -m "${master}" -F viewAngleList -r)
rm "${tempFName}"

rm "${master}"


