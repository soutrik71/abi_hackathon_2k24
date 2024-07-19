import numpy as np
import pandas as pd


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    int_types = {
        "int8": np.iinfo(np.int8),
        "int16": np.iinfo(np.int16),
        "int32": np.iinfo(np.int32),
        "int64": np.iinfo(np.int64),
    }
    float_types = {
        "float16": np.finfo(np.float16),
        "float32": np.finfo(np.float32),
        "float64": np.finfo(np.float64),
    }

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                for dtype, info in int_types.items():
                    if c_min > info.min and c_max < info.max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                for dtype, info in float_types.items():
                    if c_min > info.min and c_max < info.max:
                        df[col] = df[col].astype(dtype)
                        break
        elif col_type == "object":
            df[col] = df[col].astype("object")

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


if __name__ == "__main__":

    # Example usage
    df = pd.DataFrame(
        {
            "int_col": np.random.randint(0, 100, size=100000),
            "float_col": np.random.random(size=100000),
            "object_col": ["A"] * 50000 + ["B"] * 50000,
        }
    )

    df = reduce_mem_usage(df)
