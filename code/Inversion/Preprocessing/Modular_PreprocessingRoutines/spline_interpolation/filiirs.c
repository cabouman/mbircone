/*		mex_filiirs.c	: Matlab .MEX file
 
 M. Unser / AUG-93 

  The matlab calling syntax is:

			[y] = filiirs(x, zi, c0)

Modified by J.K. April 1998, to compile under Matlab 5 and gcc

*/

#include <math.h>
#include "mex.h"

/*************************************************************
 *	Title : filiirs.c
 *
 * PURPOSE : Symmetric infinite impulse response filter
 * of the form :
 *		H[z] = c0 * Product[ -zi[i] / ( (1-zi[i]*z) (1-zi[i]/z) ), {i,1,nz}]
 *
 * Signals are extended using mirror boundary conditions.
 *
 *  Michael Unser / BEIP	JAN-90 / AUG-93
 *
 * Input signal  : x[k], k=0,....,nx-1
 * Output signal : y[k], k=0,....,nx-1
 *
 * Note : For a unity gain, c0 should be set to
 * c0= Product[(1-zi[i])^2/-zi[i],{i,1,nz}]
 *************************************************************/
#define MIN(x,y) ( ( (x) < (y) ) ? (x) : (y) )

void filiirs(double x[],double y[],int nx,double zi[],int nz,double c0);
void filiirs(double x[],double y[],int nx,double zi[],int nz,double c0)
{
	double x0,xN,log10_epsi=-8.,xk;
	int k,iz,k0;
	   for (k=0;k<nx;k++) y[k]=c0*x[k];
	   for (iz=0;iz<nz;iz++)
	   {
	        k0=(int)(log10_epsi/log10(fabs(zi[iz])));
	        k0=MIN(k0,nx-1);
			x0=y[k0];xN=y[nx-1];
			for (k=k0-1;k>=0;k--) x0=y[k]+zi[iz]*x0;
			y[0]=x0;
			for (k=1;k<nx;k++) y[k]=y[k]+zi[iz]*y[k-1];
			y[nx-1]=-zi[iz]*(2.*y[nx-1] - xN) / (1.-zi[iz]*zi[iz]);
			for (k=nx-2;k>=0;k--) y[k]=zi[iz]*(y[k+1]-y[k]);
		}

}

/* Input Arguments */

#define	X_IN	prhs[0]
#define	Z_IN	prhs[1]
#define	C0_IN	prhs[2]

/* Output Arguments */

#define	Y_OUT	plhs[0]


#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))


#ifdef __STDC__
void mexFunction(
	int		nlhs,
	mxArray	*plhs[],
	int		nrhs,
	const mxArray	*prhs[]
	)
#else
mexFunction(nlhs, plhs, nrhs, prhs)
int nlhs, nrhs;
mxArray *plhs[], *prhs[];
#endif
{
	double	*y;
	double	*x,*zi,c0;
	unsigned int	m,n;
	int nx,nz,nmin;


	/* Check for proper number of arguments */

	if (nrhs != 3) {
		mexErrMsgTxt("FILIIRS requires three input arguments.");
	} else if (nlhs != 1) {
		mexErrMsgTxt("FILIIRS requires one output argument.");
	}


	/* Check the dimensions of input variables */

	m = mxGetM(X_IN);
	n = mxGetN(X_IN);
	nx=max(m,n);nmin=min(m,n);

	if (!mxIsNumeric(X_IN) || mxIsComplex(X_IN) || 
		mxIsSparse(X_IN)  || !mxIsDouble(X_IN) ||
		(nmin != 1)) {
		mexErrMsgTxt("FILIIRS requires that X be a vector.");
	}

	/* Create a matrix for the return argument */

	Y_OUT = mxCreateDoubleMatrix(m,n,mxREAL) ;

	m = mxGetM(Z_IN);
	n = mxGetN(Z_IN);
	nz=max(m,n);nmin=min(m,n);

	if (!mxIsNumeric(Z_IN) || mxIsComplex(Z_IN) || 
		mxIsSparse(Z_IN)  || !mxIsDouble(Z_IN) ||
		(nmin != 1)) {
		mexErrMsgTxt("FILIIRS requires that ZI be a vector.");
	}

	/* Assign pointers to the various parameters */

	y = mxGetPr(Y_OUT);
	x = mxGetPr(X_IN);
	zi  = mxGetPr(Z_IN);
	c0= (double) *mxGetPr(C0_IN);
	

	/* Do the actual computations in a subroutine */

	filiirs(x,y,nx,zi,nz,c0);
	return;
}

