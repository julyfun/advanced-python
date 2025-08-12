import zarr
import sys
from rich.tree import Tree
from rich import print

def build_tree(group, tree):
    for name, item in group.items():
        if isinstance(item, zarr.hierarchy.Group):
            subtree = tree.add(f"[bold]{name}/[/bold]")
            build_tree(item, subtree)
        else:
            dtype = item.dtype
            shape = item.shape
            tree.add(f"{name} [green]{dtype}[/green] {shape}")

store = zarr.DirectoryStore(sys.argv[1])
root = zarr.open(store, mode='r')
tree = Tree("[bold]Zarr Store[/bold]")
build_tree(root, tree)
print(tree)

