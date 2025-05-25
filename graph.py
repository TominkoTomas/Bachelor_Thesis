import matplotlib.pyplot as plt

data = [
    # non-hallucinated
    {
        "label": "non-hallucinated",
        "answer": "a dirty plate",
        "token_data": [
            {"token": " a", "logprob": -4.354257583618164},
            {"token": " dirty", "logprob": -0.566989004611969},
            {"token": " plate", "logprob": -0.06651586294174194},
        ],
    },
    {
        "label": "non-hallucinated",
        "answer": "The plates.",
        "token_data": [
            {"token": " The", "logprob": -2.159665584564209},
            {"token": " plates", "logprob": -2.160508632659912},
            {"token": ".", "logprob": -0.6810844540596008},
        ],
    },
    {
        "label": "non-hallucinated",
        "answer": "She washed a dirty plate.",
        "token_data": [
            {"token": " She", "logprob": -1.788861632347107},
            {"token": " washed", "logprob": -1.1984999179840088},
            {"token": " a", "logprob": -0.9647610187530518},
            {"token": " dirty", "logprob": -0.12195036560297012},
            {"token": " plate", "logprob": -0.07018508017063141},
            {"token": ".", "logprob": -0.4578849673271179},
        ],
    },
    {
        "label": "non-hallucinated",
        "answer": "The dishes.",
        "token_data": [
            {"token": " The", "logprob": -2.159665584564209},
            {"token": " dishes", "logprob": -0.4617094397544861},
            {"token": ".", "logprob": -0.6342569589614868},

        ],
    },

    # hallucinated
    {
        "label": "hallucinated",
        "answer": "The dirty plate",
        "token_data": [
            {"token": " The", "logprob": -1.8979122638702393},
            {"token": " dirty", "logprob": -1.7264529466629028},
            {"token": " plate", "logprob": -2.2881698608398438},
        ],
    },
    {
        "label": "hallucinated",
        "answer": "The plates",
        "token_data": [
            {"token": " The", "logprob": -1.8979122638702393},
            {"token": " plates", "logprob": -2.067864418029785},
        ],
    },
    {
        "label": "hallucinated",
        "answer": "He washed the dishes.",
        "token_data": [
            {"token": " He", "logprob": -0.9782535433769226},
            {"token": " washed", "logprob": -1.7248057126998901},
            {"token": " the", "logprob": -1.0702532529830933},
            {"token": " dishes", "logprob": -0.5193784236907959},
            {"token": ".", "logprob": -0.3618547022342682},
        ],
    },
    {
        "label": "hallucinated",
        "answer": "A dirty plate",
        "token_data": [
            {"token": " A", "logprob": -2.9817955493927},
            {"token": " dirty", "logprob": -0.7646846771240234},
            {"token": " plate", "logprob": -0.06513328850269318},
        ],
    },
]

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
colors = {'non-hallucinated': 'blue', 'hallucinated': 'red'}

for idx, entry in enumerate(data, 1):
    label = entry["label"]
    logprobs = [tok["logprob"] for tok in entry["token_data"]]
    x = list(range(1, len(logprobs) + 1))
    color = colors[label]
    answer = entry["answer"]

    # Plot line
    plt.plot(x, logprobs, marker='o', color=color, label=f"{idx}: {answer[:30]}")
    
    # Add index number at the end of each line
    plt.text(x[-1] + 0.2, logprobs[-1], str(idx), fontsize=9, color=color, fontweight='bold')

plt.title("Tokenové logaritmické pravdepodobnosti podla typu odpovede")
plt.xlabel("Index tokenu")
plt.ylabel("Logaritmická pravdepodobnosť")
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
