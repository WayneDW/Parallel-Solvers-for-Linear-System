/*! @file MatMarket_to_PETScBin.c
 *        @brief : This routine converts Matrix in matrix market file to PETSc Binary format
*
*
*  Created 4/14/2016
*/

/*PETSc Headers */
#include "petsc.h"
#include "petscmat.h"

int main(int argc,char **args)
{

  /*PETSc Mat Object */
  Mat         pMat;
  /* Input matrix market file and output PETSc binary file */
  char        inputFile[128],outputFile[128],buf[128];

  /* number rows, columns, non zeros etc */
  int         i, m,n,nnz,ierr,col,row;

  /*We compute no of nozeros per row for PETSc Mat object pre-allocation*/  
  int *nnzPtr;
  /*Maximum nonzero in nay row */
  int maxNNZperRow=0;
  /*Row number containing max non zero elements */
  int maxRowNum = 0;
  /*Just no of comments that will be ignore during successive read of file */
  int numComments=0;

  /* This is  variable of type double */
  PetscScalar val;

  /*File handle for read and write*/
  FILE*       file;
  /*File handle for writing nonzero elements distribution per row */
  FILE 	      *fileRowDist;

   /*PETSc Viewer is used for writing PETSc Mat object in binary format */
   PetscViewer view;
  /*Just record time required for conversion */
  PetscLogDouble t1,t2,elapsed_time;

  /*Initialise PETSc lib */
  PetscInitialize(&argc,&args,(char *)0,PETSC_NULL);

  /* Just record time */
  ierr = PetscGetTime(&t1); CHKERRQ(ierr);

  /*Get name of matrix market file from command line options and Open file*/
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",inputFile,127,PETSC_NULL);
  ierr = PetscFOpen(PETSC_COMM_SELF,inputFile,"r",&file);

  /* Just count the comment lines in the file */
  while(1)
  {
  	fgets(buf,128,file);
        /*If line starts with %, its a comment */
        if(buf[0] == '%')
	{
	   printf("\n IGNORING COMMENT LINE : IGNORING....");
	   numComments++; 
	}
	else
	{
	   /*Set Pointer to Start of File */
	   fseek(file, 0, SEEK_SET );
           int num = numComments;

	   /* and just move pointer to the entry in the file which indicates row nums, col nums and non zero elements */
	   while(num--)
	   	fgets(buf,128,file);
	   break;
	}
  }

  /*Reads size of sparse matrix from matrix market file */
  fscanf(file,"%d %d %d\n",&m,&n,&nnz);
  printf ("ROWS = %d, COLUMNS = %d, NO OF NON-ZEROS = %d\n",m,n,nnz);

  /*Now we will calculate non zero elelments distribution per row */
  nnzPtr = (int *) calloc (sizeof(int),  m);

  /*This is similar to calculate histogram or frequency of elements in the array */
  for (i=0; !feof(file); i++) 
  {
  	  fscanf(file,"%d %d %le\n",&row,&col,&val);
	  row = row-1; col = col-1 ;
	  nnzPtr[row]++;
  }

  printf("\n ROW DISTRIBUTION CALCULATED....WRITING TO THE FILE..!");
  fflush(stdout);

  /*Write row distribution to the file ROW_STR.dat */
  fileRowDist =  fopen ("ROW_DISTR.dat", "w");
  for (i=0; i< m; i++)
  {
     fprintf(fileRowDist, "%d\t %d\n", i, nnzPtr[i]);
     /*Find max num of of nonzero for any row of the matrix and that row number */
     if( maxNNZperRow < nnzPtr[i] )
     {	  /*store max nonzero for any row*/
	  maxNNZperRow =  nnzPtr[i];
	  /*row that contains max non zero elements*/
          maxRowNum = i; 
          
     }
  }
  /*Close File */
  fclose(fileRowDist);

  printf("\n MAX NONZERO FOR ANY ROW ARE : %d & ROW NUM IS : %d", maxNNZperRow, maxRowNum );
  
  /* Again set the file pointer the fist data record in matrix market file*
   * Note that we can directly move ponts with fseek, but as this is text file 
   * we are simple reading line by line
   */
  fseek(file, 0, SEEK_SET );
  numComments++;
  while(numComments--)
	fgets(buf,128,file);


  /* Its important to pre-allocate memory by passing max non zero for any row in the matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,maxNNZperRow,PETSC_NULL,&pMat);
  /* OR we can also pass row distribution of nozero elements for every row */
  /* ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,0,nnzPtr,&pMat);*/

  /*Now Set matrix elements values form matrix market file */
  for (i=0; i<nnz; i++) 
  {
	    /*Read matrix element from matrix market file*/
	    fscanf(file,"%d %d %le\n",&row,&col,&val);
            /*In matrix market format, rows and columns starts from 1 */
	    row = row-1; col = col-1 ;
	    /* For every non zero element,insert that value at row,col position */	
	    ierr = MatSetValues(pMat,1,&row,1,&col,&val,INSERT_VALUES);
  }
  fclose(file);
  /*Matrix Read Complete */
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n MATRIX READ...DONE!");

  /*Now assemeble the matrix */
  ierr = MatAssemblyBegin(pMat,MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(pMat,MAT_FINAL_ASSEMBLY);

  /* Now open output file for writing into PETSc Binary FOrmat*/
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",outputFile,127,PETSC_NULL);CHKERRQ(ierr);
  /*With the PETSc Viewer write output to File*/
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outputFile,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  /*Matview will dump the Mat object to binary file */
  ierr = MatView(pMat,view);CHKERRQ(ierr);

  /* Destroy the data structure */
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&pMat);CHKERRQ(ierr);

  /*Just for statistics*/
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  elapsed_time = t2 - t1;     
  ierr = PetscPrintf(PETSC_COMM_SELF,"ELAPSE TIME: %g\n",elapsed_time);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}