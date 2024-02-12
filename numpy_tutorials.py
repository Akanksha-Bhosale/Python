{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ada26bc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Numpy tutorial"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7eef7749",
   "metadata": {},
   "outputs": [],
   "source": [
    "##it is always important to import numpy initially\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f7e9eead",
   "metadata": {},
   "outputs": [],
   "source": [
    "my_list=[1,2,3,4,5,6]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "eb219723",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr=np.array(my_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "34687331",
   "metadata": {},
   "outputs": [],
   "source": [
    "##to check the type of arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7aa81cc3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "35f7623d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5 6]\n"
     ]
    }
   ],
   "source": [
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7d453af0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6,)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##to check dimension of arr\n",
    "arr.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "789ffa59",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5 6]\n"
     ]
    }
   ],
   "source": [
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "04b8d887",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "aac99282",
   "metadata": {},
   "outputs": [],
   "source": [
    "##multidimentional array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "730e3526",
   "metadata": {},
   "outputs": [],
   "source": [
    "my_ls1=[1,2,3,4,5,6]\n",
    "my_ls2=[1,3,5,7,9,11]\n",
    "my_ls3=[2,4,6,8,10,12]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a30aa012",
   "metadata": {},
   "outputs": [],
   "source": [
    "myarr=np.array([my_ls1,my_ls2,my_ls3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "8d72548c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  4,  5,  6],\n",
       "       [ 1,  3,  5,  7,  9, 11],\n",
       "       [ 2,  4,  6,  8, 10, 12]])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "0a547a69",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3  4  5  6]\n",
      " [ 1  3  5  7  9 11]\n",
      " [ 2  4  6  8 10 12]]\n"
     ]
    }
   ],
   "source": [
    "print(myarr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "beb45795",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 6)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "fb9afa49",
   "metadata": {},
   "outputs": [],
   "source": [
    "##array dimensions can be changed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5096cd4d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4,  5,  6],\n",
       "       [ 1,  3,  5],\n",
       "       [ 7,  9, 11],\n",
       "       [ 2,  4,  6],\n",
       "       [ 8, 10, 12]])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr.reshape([6,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "dcf67eeb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  4,  5,  6,  1,  3,  5,  7,  9, 11,  2,  4,  6,  8,\n",
       "        10, 12]])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr.reshape([1,18])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "98cf1eb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "##indexing\n",
    "\n",
    "##accessing array elements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "59f8ea34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  4,  5,  6],\n",
       "       [ 1,  3,  5,  7,  9, 11],\n",
       "       [ 2,  4,  6,  8, 10, 12]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "9f1494d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[1,3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "d3d5cf8c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [1, 3]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[0:2,0:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "96f40cc6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 9, 11],\n",
       "       [10, 12]])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[1:,4:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "d14cd950",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[1,-0:-5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "49338085",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  3,  5,  7,  9, 11])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "7ecb9ec5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  4,  6,  8, 10, 12])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[-1,]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ecc5478d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  3,  5,  7,  9, 11])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[-2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "48557ecc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[-0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "9953904e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3],\n",
       "       [4]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[1:4,1:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "1be422a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 5, 7, 9])"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[1,1:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "2d2b6ae6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  4,  6,  8, 10, 12])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr[-1,]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "5ccc9702",
   "metadata": {},
   "outputs": [],
   "source": [
    "##inbuilt function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "319098c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "myarr=np.arange(0,18)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "59a6f96b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
       "       17])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myarr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "5e375b6f",
   "metadata": {},
   "outputs": [],
   "source": [
    "##using step fucntion\n",
    "arr=np.arange(0,15,step=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "22a53575",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  2,  4,  6,  8, 10, 12, 14])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "d0b3c370",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.        ,  0.20408163,  0.40816327,  0.6122449 ,  0.81632653,\n",
       "        1.02040816,  1.2244898 ,  1.42857143,  1.63265306,  1.83673469,\n",
       "        2.04081633,  2.24489796,  2.44897959,  2.65306122,  2.85714286,\n",
       "        3.06122449,  3.26530612,  3.46938776,  3.67346939,  3.87755102,\n",
       "        4.08163265,  4.28571429,  4.48979592,  4.69387755,  4.89795918,\n",
       "        5.10204082,  5.30612245,  5.51020408,  5.71428571,  5.91836735,\n",
       "        6.12244898,  6.32653061,  6.53061224,  6.73469388,  6.93877551,\n",
       "        7.14285714,  7.34693878,  7.55102041,  7.75510204,  7.95918367,\n",
       "        8.16326531,  8.36734694,  8.57142857,  8.7755102 ,  8.97959184,\n",
       "        9.18367347,  9.3877551 ,  9.59183673,  9.79591837, 10.        ])"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##linspace function\n",
    "np.linspace(0,10,50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "8ddab2ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "##copy function and broadcasting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "0f852662",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr1=np.array([1,2,3,4,5,6,7,8,9,10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "147cbf7a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "acf2c5c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr1[3:]=100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "9b4bdd97",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  1,   2,   3, 100, 100, 100, 100, 100, 100, 100])"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "7d4457d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr11=arr1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "62716083",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr11[3:]=500"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "604afe93",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  1,   2,   3, 500, 500, 500, 500, 500, 500, 500])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "3337513f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  1,   2,   3, 500, 500, 500, 500, 500, 500, 500])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "49eeb068",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr11=arr.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "f8eea033",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0  2  4  6  8 10 12 14]\n"
     ]
    }
   ],
   "source": [
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "74d720cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[   1    2    3 1000 1000 1000 1000 1000 1000 1000]\n",
      "[   0    2    4 1000 1000 1000 1000 1000]\n"
     ]
    }
   ],
   "source": [
    "print(arr1)\n",
    "arr11[3:]=1000\n",
    "print(arr11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "32ad9ed8",
   "metadata": {},
   "outputs": [],
   "source": [
    "##some conditions very useful  in exploratory data analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "ca4b032e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([   2,    4,    6, 2000, 2000, 2000, 2000, 2000, 2000, 2000])"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1*2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "d7a63b32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  0.5,   1. ,   1.5, 500. , 500. , 500. , 500. , 500. , 500. ,\n",
       "       500. ])"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1/2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "44857489",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True, False, False, False, False, False, False,\n",
       "       False])"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "val=2\n",
    "arr1<10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "d3bfe183",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Create arrays and reshape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "80c59d0c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1],\n",
       "       [2, 3],\n",
       "       [4, 5],\n",
       "       [6, 7],\n",
       "       [8, 9]])"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.arange(0,10).reshape(5,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "c03072c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "myyy=np.arange(0,10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "d38785a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myyy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "841b1a6a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2, 3, 4],\n",
       "       [5, 6, 7, 8, 9]])"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myyy.reshape(2,5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "e8b03e68",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 1., 1., 1., 1., 1.])"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##ones function\n",
    "np.ones(6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "8cfaf85e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 1],\n",
       "       [1, 1],\n",
       "       [1, 1],\n",
       "       [1, 1],\n",
       "       [1, 1]])"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.ones((5,2),dtype=int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "c86a941e",
   "metadata": {},
   "outputs": [],
   "source": [
    "##random distributiom"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "c7209f36",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.50208902, 0.90533162, 0.08490028],\n",
       "       [0.27192858, 0.06555278, 0.07390925],\n",
       "       [0.70766018, 0.8180646 , 0.73068371]])"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.rand(3,3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "d8291bb5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.63872693,  0.4641811 ,  2.18634578,  1.91097236],\n",
       "       [ 0.38483743, -0.03169823, -1.49269927,  0.43503976],\n",
       "       [-0.04532722,  1.61234803,  0.76514108, -1.23971868],\n",
       "       [-2.52074852, -0.47978882, -1.50302595,  0.41277979]])"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.randn(4,4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "1347626c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[72, 18, 66, 19],\n",
       "       [93, 89, 39, 11]])"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.randint(0,100,8).reshape(2,4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "d46d235f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.98810934, 0.97544018, 0.7057453 , 0.12703149, 0.70822948]])"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.random_sample((1,5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab7375a2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
