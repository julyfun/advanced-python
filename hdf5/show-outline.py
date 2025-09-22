import h5py
import sys

def show_hdf5_outline(file_path):
    """显示HDF5文件的内容大纲"""
    def print_attrs(name, obj):
        """递归打印HDF5对象的属性和内容"""
        # 获取缩进级别（用于格式化输出）
        indent = name.count('/') * '  '
        
        # 打印组或数据集的名称
        if isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            # 获取数据集的形状和数据类型
            shape = obj.shape
            dtype = obj.dtype
            print(f"{indent}Dataset: {name} (Shape: {shape}, Type: {dtype})")
        
        # 打印对象的属性（如果有）
        if obj.attrs:
            attrs_indent = indent + '  '
            print(f"{attrs_indent}Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{attrs_indent}  {attr_name}: {attr_value}")
    
    try:
        # 打开HDF5文件
        with h5py.File(file_path, 'r') as f:
            print(f"HDF5 File: {file_path}")
            print("Content Outline:")
            # 递归遍历并打印文件内容
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # 替换为你的HDF5文件路径
    show_hdf5_outline(sys.argv[1])
