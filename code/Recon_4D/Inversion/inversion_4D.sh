#!/usr/bin/env bash

generalError()
{
	>&2 echo "inversion 4D error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


if [[ $# -ne 2 ]]; then
	messageError "number of inputs not 2"
	generalError "$0 $@"
	exit 1
fi

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")

master_4D=$(readlink -f ${1})
mode=${2}

invMasterList=$(bash $plainParamsFile -a get -m $master_4D -F invMasterList -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


read -r num_vols < $invMasterList
echo "num_vols = $num_vols"

cd ../../Inversion/run
for (( vol_id=1; vol_id<=$num_vols; vol_id++ ))
do
	echo "Inversion time: $vol_id of $num_vols "
	line_num=$(( vol_id + 1 ))
	masterFile_temp="`sed -n ${line_num}p $invMasterList`"
	echo "mode: $mode"

	if [[ ${mode} = "reuseLastRecon" ]]; then
		
		if [[  $vol_id == 1 ]]; then
			# start first time point
			echo "Inversion with masterfile: ${masterFile_temp}"
			./runall.sh $masterFile_temp
		else
			# update by previous time point

			echo "Inversion: Generate System Matrix with masterfile: ${masterFile_temp}"
			./genSysMatrix.sh $masterFile_temp

			echo "Copying Recon of last time point"
			recon_old=$(bash $plainParamsFile -a get -m $masterFile_temp_old -F binaryFNames -f recon  -r)
			if [[  $? != 0 ]]; then messageError "getting recon old from $masterFile_temp_old " ; generalError "$0 $@"; exit 1; fi
			recon_new=$(bash $plainParamsFile -a get -m $masterFile_temp -F binaryFNames -f phantom  -r)
			if [[  $? != 0 ]]; then messageError "getting recon from $masterFile_temp " ; generalError "$0 $@"; exit 1; fi
			cp -r $recon_old $recon_new
			if [[  $? != 0 ]]; then messageError "copying $recon_old to $recon_new " ; generalError "$0 $@"; exit 1; fi

			echo "Inversion: Init with masterfile: ${masterFile_temp}"
			$(bash $plainParamsFile -a set -m $masterFile_temp -F reconParams -f isPhantomPresent -v 1)
			$(bash $plainParamsFile -a set -m $masterFile_temp -F reconParams -f isUsePhantomToInitErrSino -v 1)
			./initialize.sh $masterFile_temp
			
			echo "Inversion: Recon with masterfile: ${masterFile_temp}"
			./Recon.sh $masterFile_temp	

			# echo "Inversion with masterfile: ${masterFile_temp}"
			# ./runall.sh $masterFile_temp
		fi

	fi

	if [[ ${mode} = "initAll_with_FirstRecon" ]]; then


		if [[  $vol_id == 1 ]]; then
			masterFile1=${masterFile_temp}
		fi

		echo "Copying Recon to phantom"
		recon=$(bash $plainParamsFile -a get -m $masterFile1 -F binaryFNames -f recon  -r)
		if [[  $? != 0 ]]; then messageError "getting recon old from $masterFile1 " ; generalError "$0 $@"; exit 1; fi
		phantom=$(bash $plainParamsFile -a get -m $masterFile_temp -F binaryFNames -f phantom  -r)
		if [[  $? != 0 ]]; then messageError "getting recon from $masterFile_temp " ; generalError "$0 $@"; exit 1; fi
		cp -r $recon $phantom
		if [[  $? != 0 ]]; then messageError "copying $recon to $phantom " ; generalError "$0 $@"; exit 1; fi

		echo "Inversion: Init with masterfile: ${masterFile_temp}"
		$(bash $plainParamsFile -a set -m $masterFile_temp -F reconParams -f isPhantomPresent -v 1)
		$(bash $plainParamsFile -a set -m $masterFile_temp -F reconParams -f isUsePhantomToInitErrSino -v 1)
		./initialize.sh $masterFile_temp

	fi

	if [[ ${mode} = "initAll_with_Recon" ]]; then

		echo "Copying Recon to phantom"
		recon=$(bash $plainParamsFile -a get -m $masterFile_temp -F binaryFNames -f recon  -r)
		if [[  $? != 0 ]]; then messageError "getting recon old from $masterFile_temp " ; generalError "$0 $@"; exit 1; fi
		phantom=$(bash $plainParamsFile -a get -m $masterFile_temp -F binaryFNames -f phantom  -r)
		if [[  $? != 0 ]]; then messageError "getting recon from $masterFile_temp " ; generalError "$0 $@"; exit 1; fi
		cp -r $recon $phantom
		if [[  $? != 0 ]]; then messageError "copying $recon to $phantom " ; generalError "$0 $@"; exit 1; fi

		echo "Inversion: Init with masterfile: ${masterFile_temp}"
		$(bash $plainParamsFile -a set -m $masterFile_temp -F reconParams -f isPhantomPresent -v 1)
		$(bash $plainParamsFile -a set -m $masterFile_temp -F reconParams -f isUsePhantomToInitErrSino -v 1)
		./initialize.sh $masterFile_temp

	fi

	if [[ ${mode} = "genSysMatrix" ]]; then
		
		echo "Generate System Matrix with masterfile: ${masterFile_temp}"
		./genSysMatrix.sh $masterFile_temp

	fi

	if [[ ${mode} = "runall" ]]; then
		
		echo "Inversion with masterfile: ${masterFile_temp}"
		./runall.sh $masterFile_temp

	fi

	if [[ ${mode} = "Recon" ]]; then
		
		echo "Inversion with masterfile: ${masterFile_temp}"
		./Recon.sh $masterFile_temp

	fi
	

	masterFile_temp_old=$masterFile_temp

done

cd "${scriptDir}"

