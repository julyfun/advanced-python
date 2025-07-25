from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    @classmethod
    def from_json(cls, json_str: str) -> "Config":
        data = json.loads(json_str)
        # 使用字典解包和默认参数，缺少的键会自动使用默认值
        return cls(**data)

# 示例
json_data = '{"host": "example.com", "debug": true}'
config = Config.from_json(json_data)
print(config)  # 输出: Config(host='example.com', port=8080, debug=True)

