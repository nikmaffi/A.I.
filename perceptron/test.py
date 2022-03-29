import neuron

def main():
    func = {
        "inputs": [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        "results": [0, 0, 0, 0, 0, 0, 0, 1]
    }

    n = neuron.Neuron(3, 10000, 0.17)

    n.fit(func["inputs"], func["results"])

    for i in func["inputs"]:
        res = n.active(i)
        print(f"{i[0]} {i[1]} {i[2]} | {round(res)} \t {res}")

if __name__ == "__main__":
    main()