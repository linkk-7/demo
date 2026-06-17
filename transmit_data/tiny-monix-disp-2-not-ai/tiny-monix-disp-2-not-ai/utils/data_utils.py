
from dao.base_dao import T
import copy


def copy_properties(source: T, target: T, deepcopy: bool = True) -> None:
    """
    将source中的非空字段复制到target中，是否需要深拷贝：deepcopy
    """
    source_dict = source.model_dump(exclude_none=True)
    print("\n\nsource_dict", source_dict, "\n\n")
    for field, value in source_dict.items():
        if deepcopy:
            setattr(target, field, copy.deepcopy(value))
        else:
            setattr(target, field, value)            
    # target.model_copy(update=source_dict, deep=True)