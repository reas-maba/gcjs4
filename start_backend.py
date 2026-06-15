import subprocess
import sys
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print('🐕 DogBreeds Recognizer V1.0')
    print('=' * 40)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'backend/requirements.txt'
        ], check=True)
        print('✅ 依赖安装完成')
    except subprocess.CalledProcessError:
        print('❌ 依赖安装失败')
        return
    
    print('🚀 启动后端服务...')
    subprocess.run([
        sys.executable, 'backend/src/app.py'
    ])

if __name__ == '__main__':
    main()