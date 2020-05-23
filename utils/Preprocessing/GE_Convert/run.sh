

if [ $# == 1 ]; then
	echo "File specified: ${1}"
	fName_rel=${1}
	fName_abs=$(readlink -f ${fName_rel})
else
	echo "Error: enter path to .vol file as argument"
	exit 1
fi

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

matlab -nodesktop -nosplash -r "GE_convertVolData('${fName_abs}'); quit"

