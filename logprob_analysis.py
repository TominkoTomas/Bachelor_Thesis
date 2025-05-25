import pandas as pd
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def parse_token_data(token_data_str):
    """Parsovanie JSON reťazca token_data a extrahovanie logprobs"""
    try:
        token_data = json.loads(token_data_str)
        logprobs = [token['logprob'] for token in token_data]
        return logprobs
    except:
        return []

def calculate_metrics(logprobs):
    """Výpočet rôznych metrík pre logprobs"""
    if not logprobs:
        return None
    
    return {
        'mean_logprob': np.mean(logprobs),
        'sum_logprob': np.sum(logprobs),
        'weighted_avg': np.sum(logprobs) / len(logprobs),  # Rovnaké ako mean, ale explicitné
        'length_weighted': np.sum(logprobs) / np.sqrt(len(logprobs)),  # Úprava podľa dĺžky
        'token_count': len(logprobs),
        'min_logprob': np.min(logprobs),
        'max_logprob': np.max(logprobs),
        'std_logprob': np.std(logprobs)
    }

def create_violin_plots(results_df):
    """Vytvorenie husľových grafov pre distribúcie logprob"""
    # Nastavenie štýlu grafov
    plt.style.use('default')
    
    # Vytvorenie obrázka s podgrafmi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Príprava dát na vykreslenie
    plot_data = []
    for _, row in results_df.iterrows():
        plot_data.append({
            'mean_logprob': row['mean_logprob'],
            'length_weighted': row['length_weighted'],
            'category': 'Halucinačná' if row['is_hallucinated'] == 1 else 'Nehalucinačná'
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Definovanie farieb s explicitným mapovaním
    color_map = {'Halucinačná': '#FF4444', 'Nehalucinačná': '#4444FF'}  # Červená pre halucinačné, modrá pre nehalucinačné
    
    # Graf 1: Distribúcia priemerných logprob
    violin1 = sns.violinplot(data=plot_df, x='category', y='mean_logprob', ax=ax1, palette=color_map, order=['Halucinačná', 'Nehalucinačná'])
    ax1.set_title('Distribúcia priemerov logaritmických pravdepodobností', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Typ odpovede', fontsize=12)
    ax1.set_ylabel('Priemerná logaritmická pravdepodobnosť', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Pridanie značiek priemerov s bielym pozadím pre viditeľnosť
    hall_mean = plot_df[plot_df['category'] == 'Halucinačná']['mean_logprob'].mean()
    non_hall_mean = plot_df[plot_df['category'] == 'Nehalucinačná']['mean_logprob'].mean()
    
    # Pridanie značiek priemerov ako veľké biele kruhy s farebnými okrajmi
    ax1.plot(0, hall_mean, marker='o', markersize=12, color='white', 
             markeredgecolor='red', markeredgewidth=3, label=f'Hal. priemer: {hall_mean:.3f}', zorder=10)
    ax1.plot(1, non_hall_mean, marker='o', markersize=12, color='white', 
             markeredgecolor='blue', markeredgewidth=3, label=f'Nehal. priemer: {non_hall_mean:.3f}', zorder=10)
    
    # Pridanie textových anotácií pre priemery s pozaďovými rámčekmi
    ax1.text(0.15, hall_mean, f'{hall_mean:.3f}', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
    ax1.text(1.15, non_hall_mean, f'{non_hall_mean:.3f}', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='blue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.9))
    
    # Graf 2: Distribúcia logprob upravených podľa dĺžky
    violin2 = sns.violinplot(data=plot_df, x='category', y='length_weighted', ax=ax2, palette=color_map, order=['Halucinačná', 'Nehalucinačná'])
    ax2.set_title('Distribúcia logaritmických pravdepodobností upravených podľa dĺžky', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Typ odpovede', fontsize=12)  
    ax2.set_ylabel('Logaritmická pravdepodobnosť upravená podľa dĺžky', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Pridanie značiek priemerov pre druhý graf
    hall_length_mean = plot_df[plot_df['category'] == 'Halucinačná']['length_weighted'].mean()
    non_hall_length_mean = plot_df[plot_df['category'] == 'Nehalucinačná']['length_weighted'].mean()
    
    # Pridanie značiek priemerov ako veľké biele kruhy s farebnými okrajmi
    ax2.plot(0, hall_length_mean, marker='o', markersize=12, color='white', 
             markeredgecolor='red', markeredgewidth=3, zorder=10)
    ax2.plot(1, non_hall_length_mean, marker='o', markersize=12, color='white', 
             markeredgecolor='blue', markeredgewidth=3, zorder=10)
    
    # Pridanie textových anotácií pre priemery s pozaďovými rámčekmi
    ax2.text(0.15, hall_length_mean, f'{hall_length_mean:.3f}', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
    ax2.text(1.15, non_hall_length_mean, f'{non_hall_length_mean:.3f}', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='blue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.9))
    
    # Pridanie legendy na vysvetlenie farebnej schémy
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF4444', label='Halucinačná'),
                      Patch(facecolor='#4444FF', label='Nehalucinačná')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Urobenie miesta pre legendu
    plt.savefig('logprob_violin_plots_sk.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Tlač súhrnných štatistík pre grafy
    print("\n" + "="*60)
    print("SÚHRNNÉ ŠTATISTIKY HUSĽOVÝCH GRAFOV")
    print("="*60)
    
    print("\nDISTRIBÚCIE PRIEMERNÝCH LOGPROB:")
    print("-" * 40)
    for category in ['Halucinačná', 'Nehalucinačná']:
        data = plot_df[plot_df['category'] == category]['mean_logprob']
        print(f"{category}:")
        print(f"  Priemer: {data.mean():.4f}")
        print(f"  Medián: {data.median():.4f}")
        print(f"  Smer. odch.: {data.std():.4f}")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Počet: {len(data)}")
        print()
    
    print("DISTRIBÚCIE LOGPROB UPRAVENÝCH PODĽA DĹŽKY:")
    print("-" * 40)
    for category in ['Halucinačná', 'Nehalucinačná']:
        data = plot_df[plot_df['category'] == category]['length_weighted']
        print(f"{category}:")
        print(f"  Priemer: {data.mean():.4f}")
        print(f"  Medián: {data.median():.4f}")
        print(f"  Smer. odch.: {data.std():.4f}")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Počet: {len(data)}")
        print()

def main():
    # Čítanie CSV súboru
    print("Čítanie CSV súboru...")
    df = pd.read_csv(r'data/answers_final.csv')
    
    print(f"Celkový počet riadkov v datasete: {len(df)}")
    
    # Filtrovanie iba trénovacích dát (use_for_training = 1)
    training_df = df[df['use_for_training'] == 1].copy()
    print(f"Riadky s use_for_training = 1: {len(training_df)}")
    
    # Parsovanie token dát a výpočet metrík
    print("\nParsovanie token dát a výpočet metrík...")
    
    results = []
    for idx, row in training_df.iterrows():
        logprobs = parse_token_data(row['token_data'])
        metrics = calculate_metrics(logprobs)
        
        if metrics:
            result = {
                'story_id': row['story_id'],
                'character': row['character'],
                'run_number': row['run_number'],
                'is_hallucinated': row['is_hallucinated'],
                'answer': row['answer'][:50] + "..." if len(row['answer']) > 50 else row['answer'],
                **metrics
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Rozdelenie halucinačných vs nehalucinačných
    hallucinated = results_df[results_df['is_hallucinated'] == 1]
    non_hallucinated = results_df[results_df['is_hallucinated'] == 0]
    
    print(f"\nHalucinačné odpovede (is_hallucinated=1): {len(hallucinated)}")
    print(f"Nehalucinačné odpovede (is_hallucinated=0): {len(non_hallucinated)}")
    
    # Vytvorenie husľových grafov
    print("\nGenerovanie husľových grafov...")
    create_violin_plots(results_df)
    
    # Výpočet súhrnných štatistík
    print("\n" + "="*60)
    print("VÝSLEDKY ANALÝZY LOGPROB")
    print("="*60)
    
    metrics_to_compare = ['mean_logprob', 'sum_logprob', 'length_weighted', 'token_count']
    metric_names = {
        'mean_logprob': 'PRIEMERNÁ LOGPROB',
        'sum_logprob': 'SÚČET LOGPROB',
        'length_weighted': 'LOGPROB UPRAVENÁ PODĽA DĹŽKY',
        'token_count': 'POČET TOKENOV'
    }
    
    for metric in metrics_to_compare:
        print(f"\n{metric_names[metric]}:")
        print("-" * 40)
        
        hall_values = hallucinated[metric].values
        non_hall_values = non_hallucinated[metric].values
        
        hall_mean = np.mean(hall_values)
        non_hall_mean = np.mean(non_hall_values)
        
        print(f"Halucinačné (priemer ± smer.odch.):     {hall_mean:.4f} ± {np.std(hall_values):.4f}")
        print(f"Nehalucinačné (priemer ± smer.odch.):   {non_hall_mean:.4f} ± {np.std(non_hall_values):.4f}")
        print(f"Rozdiel (hal - nehal):                  {hall_mean - non_hall_mean:.4f}")
        
        # Test štatistickej významnosti
        try:
            t_stat, p_value = stats.ttest_ind(hall_values, non_hall_values)
            print(f"T-test p-hodnota:                       {p_value:.6f}")
            print(f"Štatisticky významný:                   {'Áno' if p_value < 0.05 else 'Nie'}")
        except:
            print("Nebolo možné vykonať t-test")
        
        # Veľkosť efektu (Cohenovo d)
        pooled_std = np.sqrt(((len(hall_values)-1)*np.var(hall_values) + 
                             (len(non_hall_values)-1)*np.var(non_hall_values)) / 
                            (len(hall_values) + len(non_hall_values) - 2))
        if pooled_std > 0:
            cohens_d = (hall_mean - non_hall_mean) / pooled_std
            print(f"Veľkosť efektu (Cohenovo d):            {cohens_d:.4f}")
    
    # Podrobný popis
    print("\n" + "="*60)
    print("PODROBNÝ POPIS")
    print("="*60)
    
    print("\nPRÍKLADY HALUCINAČNÝCH ODPOVEDÍ:")
    print("-" * 40)
    for _, row in hallucinated.iterrows():
        print(f"Príbeh {row['story_id']}, Run {row['run_number']}: Priemerná logprob = {row['mean_logprob']:.4f}, "
              f"Tokeny = {row['token_count']}, Odpoveď: {row['answer']}")
    
    print("\nPRÍKLADY NEHALUCINAČNÝCH ODPOVEDÍ:")
    print("-" * 40)
    for _, row in non_hallucinated.iterrows():
        print(f"Príbeh {row['story_id']}, Run {row['run_number']}: Priemerná logprob = {row['mean_logprob']:.4f}, "
              f"Tokeny = {row['token_count']}, Odpoveď: {row['answer']}")
    
    # Súhrn kľúčových zistení
    print("\n" + "="*60)
    print("KĽÚČOVÉ ZISTENIA")
    print("="*60)
    
    hall_mean_logprob = np.mean(hallucinated['mean_logprob'])
    non_hall_mean_logprob = np.mean(non_hallucinated['mean_logprob'])

    print(f"\nPorovnanie:")
    print(f"Halucinačné:     {hall_mean_logprob:.4f}")
    print(f"Nehalucinačné:   {non_hall_mean_logprob:.4f}")

    if hall_mean_logprob < non_hall_mean_logprob:
        print("✓ Halucinačné odpovede majú NIŽŠIE (horšie) priemerné logprob ako nehalucinačné")
        print(f"  Rozdiel: {hall_mean_logprob - non_hall_mean_logprob:.4f}")
    else:
        print("✗ Halucinačné odpovede majú VYŠŠIE (lepšie) priemerné logprob ako nehalucinačné")
        print(f"  Rozdiel: {hall_mean_logprob - non_hall_mean_logprob:.4f}")
    
    # Porovnanie upravené podľa dĺžky
    hall_length_adj = np.mean(hallucinated['length_weighted'])
    non_hall_length_adj = np.mean(non_hallucinated['length_weighted'])
    
    print(f"\nPorovnanie upravené podľa dĺžky:")
    print(f"Halucinačné:     {hall_length_adj:.4f}")
    print(f"Nehalucinačné:   {non_hall_length_adj:.4f}")
    print(f"Rozdiel:         {hall_length_adj - non_hall_length_adj:.4f}")

if __name__ == "__main__":
    main()