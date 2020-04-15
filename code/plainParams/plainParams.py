import subprocess

#      if get_set = 'get' then
#          - contents of value will be ignored and
#          - read data will be returned
#
#      if get_set = 'set' then
#          - contents of value are used to set params file and
#          - returned data is not specified
#
#      Any argument = '' will be ignored.
#
# This file need to be in the same folder as the plainParams.sh executable!

# to test run:
# plainParams('plainParams.sh', 'get', '../../control/ConeBeam/master.txt', 'preprocessingParams', 'N_beta_all', '', '')   


def plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag):

	command = 'bash '+executablePath;

	if get_set != '':
		command = command+' -a '+'"'+get_set+'"'
		if get_set=='get':
			value = ''

	if masterFile != '':
		command = command+' -m '+'"'+masterFile+'"'

	if masterField != '':
		command = command+' -F '+'"'+masterField+'"'

	if subField != '':
		command = command+' -f '+'"'+subField+'"'

	if value != '':
		command = command+' -v '+'"'+value+'"'

	if resolveFlag != '':
		command = command+' '+resolveFlag
		
		
	# print(command)
	pout = subprocess.check_output([command], shell=True)
	return pout.rstrip()

if __name__=='__main__':
	str = plainParams('plainParams.sh', 'get', '../../control/ConeBeam/master.txt', 'preprocessingParams', 'backgroundPatchLimits', '', '')   
	print(str)

