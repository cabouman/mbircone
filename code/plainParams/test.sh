

############################################
get_set="get"
masterFileName="Control/master.txt"
masterField="binaryFNames"
subField="sino"
value=""
resolve="true"


# value=$(./plainParams.sh -a $get_set -m "$masterFileName" -F "$masterField" )
value=$(bash plainParams.sh -a $get_set -m "$masterFileName" -F "$masterField" )
if [[  $? != 0 ]]; then echo "ERROR"; exit 1; fi
echo "${value}"
