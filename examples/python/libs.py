import sys
import os 
import glob


    
if __name__ == '__main__':
    path = os.path.dirname(sys.executable).replace('\\','/')
    libs = glob.glob(path + '/libs/python*')
    lib = os.path.splitext(os.path.basename(libs[-1]))[0]
    print(f'{path+"/include"},{path+"/libs"},{lib}')