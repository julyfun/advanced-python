添加以下接口:
```python
parser.add_argument('--load', type=str, help='Load checkpoint path')
parser.add_argument('--save', type=str, help='Save checkpoint path')
parser.add_argument('--train', type=int, help='Number of training epochs')
parser.add_argument('--eval', action='store_true', help='Evaluate on test set')
```
