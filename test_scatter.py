import numpy as np
try:
    data = np.load('aggregate_states.npz')
    final = data['final_states']
    initial = data['initial_states']
    outcomes = data['outcomes']
    print(f'Total trials loaded: {len(outcomes)}')
    
    hits = np.where(outcomes == 1)[0]
    reflected = np.where(outcomes == -1)[0]
    in_flight = np.where(outcomes == 0)[0]
    
    print(f'Hits: {len(hits)}, Reflected: {len(reflected)}, In Flight: {len(in_flight)}')
    
    if len(reflected) > 0:
        r_final = final[reflected]
        r_init = initial[reflected]
        print(f'Sample Reflected Final X: {r_final[:5, 1]}')
        print(f'Sample Reflected Final Y: {r_final[:5, 2]}')
        print(f'Sample Reflected Final Z: {r_final[:5, 3]}')
        print(f'Sample Reflected Initial Y: {r_init[:5, 0]}')
        
    if len(in_flight) > 0:
        i_final = final[in_flight]
        print(f'Sample In-Flight Final X: {i_final[:5, 1]}')
        print(f'Sample In-Flight Final Y: {i_final[:5, 2]}')
        print(f'Sample In-Flight Final Z: {i_final[:5, 3]}')
except Exception as e:
    print('Not ready yet:', e)
