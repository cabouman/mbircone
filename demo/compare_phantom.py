import numpy as np

def read_ND(filePath, n_dim, dtype='float32', ntype='int32'):

    with open(filePath, 'rb') as fileID:

        sizesArray = np.fromfile( fileID, dtype=ntype, count=n_dim)
        numElements = np.prod(sizesArray)
        dataArray = np.fromfile(fileID, dtype=dtype, count=numElements).reshape(sizesArray)

    return dataArray


def main():

	fname_img = 'inversion/object.recon'
	fname_ref = 'inversion/object.phantom.recon'
	
	img = read_ND(fname_img, 3)
	ref = read_ND(fname_ref, 3)

	rmse_val = np.sqrt(np.mean((img-ref)**2))
	print("RMSE between reconstruction and phantom: {}".format(rmse_val))

	norm_mse_val = np.sqrt(np.mean((img-ref)**2)/np.mean(np.maximum(img,ref)**2))
	print("Normalized MSE between reconstruction and phantom: {}".format(norm_mse_val))

if __name__ == '__main__':
	main()