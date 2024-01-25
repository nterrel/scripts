counts_dict = {}
for key, entry in df.items():
    count = sum(val == 1 for val in entry['relative_stdev'])
    counts_dict[key] = count
for key, count in counts_dict.items():
    if count > 0:
        print(f'Key: {key}, Count of ones: {count}')


filtered_dict = {}
for key, entry in output_dict.items():
    if any(val == 1 for val in entry['relative_stdev']):
        filtered_dict[key] = entry
df = pd.DataFrame.from_dict(filtered_dict, orient='index')
df.to_pickle('filtered_1x_first_w_stdev.pkl')
print(len(df))


output_dict = {}
with ds.keep_open('r') as read_ds:
    pbar = tqdm(total=ds.num_conformers)
    for group, j, conformer in read_ds.chunked_items(max_size=500):
        #print(group)
        species = conformer['species'].to(device)
        coordinates = conformer['coordinates'].to(device)
        ani_input = (species, coordinates)
        rel_range = model.force_qbc(ani_input).relative_range.squeeze()
        rel_range = [1 if i > 1 else 0 for i in rel_range]
        ani_output = (species, coordinates, rel_range)
        print(ani_output)
        output_dict[group] = {
            'species': species,
            'coordinates': coordinates,
            'relative_range': rel_range,
            'relative_stdev': rel_stdev
        }
        pbar.update()
print(output_dict)


