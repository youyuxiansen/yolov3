import os

import glob


def find_release_in_dir(dir):
	# dir: string
	ret = []
	# find 'Release'
	if 'Release' not in dir:
		pass
	else:
		ret.append(dir + 'Release')

	# find in dir
	if os.listdir(dir):
		return ret
	else:
		for dir1 in os.listdir(dir):
			ret += find_release_in_dir(dir1)
		return ret


if __name__ == '__main__':
	Release_dir = []
	lib_files = []
	for dirpath, dirnames, filename in os.walk('/Users/felix/Documents/CPPProject/grpc/.build/third_party/'):
		if 'Release' != dirpath.split('/')[-1]:
			pass
		else:
			lib_file = [file for file in filename if '.lib' in file]
			if lib_file:
				Release_dir.append(dirpath)
				lib_files += lib_file
	[dir.replace('/', '\\') for dir in Release_dir]
	print(','.join(Release_dir))
	print(';'.join(lib_files))


