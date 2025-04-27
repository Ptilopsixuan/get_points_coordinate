from mylib import function

if __name__ == "__main__":
    colors = {
        "red": [">", "<", "<", 250, 255, 10],
        "blue": ["<", "<", ">", 10, 255, 250],
    }
    function.detect(path=".\\original", type=".png", colors=colors, min_distance=25, visualize=True)
