import random
MAX_ACTIONS = 64

le_featurewise_enable_dyn_action_fallback = True
le_featurewise_enable_action_fallback = True
le_featurewise_bw_acc_check_level = 1
le_featurewise_acc_thresh = 2

le_featurewise_active_features = [0,10]
le_featurewise_feature_weights=[1.00,1.00]
le_featurewise_weight_gradient = 0.001
le_featurewise_num_tilings = [3,3]
le_featurewise_num_tiles = [128,128]
le_featurewise_hash_types = [2,2]
le_featurewise_enable_tiling_offset = [1,1]
le_featurewise_pooling_type = 2
le_featurewise_max_q_thresh = 0.50
le_featurewise_enable_action_fallback = True
le_featurewise_enable_dynamic_weight = False
le_featurewise_selective_update = False


tiling_offset = [0xaca081b9,0x666a1c67,0xc11d6a53,0x8e5d97c1,0x0d1cad54,0x874f71cb,0x20d2fa13,0x73f7c4a7,
									0x0b701f6c,0x8388d86d,0xf72ac9f2,0xbab16d82,0x524ac258,0xb5900302,0xb48ccc72,0x632f05bf,
									0xe7111073,0xeb602af4,0xf3f29ebb,0x2a6184f2,0x461da5da,0x6693471d,0x62fd0138,0xc484efb3,
									0x81c9eeeb,0x860f3766,0x334faf86,0x5e81e881,0x14bc2195,0xf47671a8,0x75414279,0x357bc5e0]


DELTA_BITS = 7

FeatureType = {'F_PC' : 0, 'F_Offset' : 1, 'F_Delta' : 2, 'F_Address':3, 
'F_PC_Offset':4, 'F_PC_Address':5, 'F_PC_Page':6, 'F_PC_Path':7, 'F_Delta_Path':8,
'F_Offset_Path':9, 'F_PC_Delta':10, 'F_PC_Offset_Delta':11, 'F_Page':12, 
'F_PC_Path_Offset':13, 'F_PC_Path_Offset_Path':14, 'F_PC_Path_Delta':15,
'F_PC_Path_Delta_Path':16, 'F_PC_Path_Offset_Path_Delta_Path':17,
'F_Offset_Path_PC':18, 'F_Delta_Path_PC':19, 'NumFeatureTypes':20}

RewardType = {'none' : 0, 'incorrect' : 1, 'correct_untimely' : 2, 'correct_timely' : 3,'out_of_bounds' : 4, 'tracker_hit' : 5, 'num_rewards' : 6}

class FeatureKnowledge:
	def __init__(self, feature_type, alpha, gamma, actions, weight, weight_gradient, num_tilings, num_tiles, zero_init, hash_type, enable_tiling_offset):
		self.m_feature_type = feature_type
		self.m_alpha = alpha
		self.m_gamma = gamma
		self.m_actions =  actions
		self.m_weight = weight
		self.m_weight_gradient = weight_gradient  
		self.m_hash_type = hash_type
		self.m_enable_tiling_offset = enable_tiling_offset
		self.m_num_tilings = num_tilings 
		self.m_num_tiles = num_tiles
		self.min_weight = 1000000
		self.max_weight = 0

		m_init_value =  None
		if zero_init:
			m_init_value = 0 
		else:
			m_init_value = 1.0/(1-gamma)

		self.m_qtable = [[[m_init_value]*self.m_actions] * self.m_num_tiles] * self.m_num_tilings


	def retrieveQ(self, state, action):
		tile_index = 0
		q_value = 0.0

		for tiling in range(self.m_num_tilings):
			tile_index = self.get_tile_index(tiling, state)
			q_value += self.m_qtable[tiling][tile_index][action]

		return self.m_weight * q_value


	def get_tile_index(self, tiling, state):
		pc = state.pc 
		page = state.page 
		address = state.address
		offset = state.offset 
		delta = state.delta 
		delta_path = state.local_delta_sig2
		pc_path = state.local_pc_sig 
		offset_path = state.local_offset_sig 

		if self.m_feature_type == 0:
			return self.process_PC(tiling, pc)
		elif self.m_feature_type == 10:
			return self.process_PC_delta(tiling, pc, delta)
		else:
			assert False


	def process_PC(self, tiling, pc):
		raw_index = self.folded_xor(pc, 2)
		if self.m_enable_tiling_offset:
			raw_index = raw_index ^ tiling_offset[tiling]
		hashed_index = self.getHash(self.m_hash_type, raw_index)
		return (hashed_index % self.m_num_tiles)


	def process_PC_delta(self, tiling, pc, delta):
		unsigned_delta = delta 
		if delta < 0:
			unsigned_delta = (((-1) * delta) + (1 << (DELTA_BITS - 1)))
		tmp = pc 
		tmp = tmp << 7
		tmp += unsigned_delta 
		raw_index = self.folded_xor(tmp, 2)
		if self.m_enable_tiling_offset:
			raw_index = raw_index ^ tiling_offset[tiling] 
		hashed_index = self.getHash(self.m_hash_type, raw_index)
		return (hashed_index % self.m_num_tiles)


	def getHash(self, selector, key):
		if selector == 2:
			return self.jenkins(key)
		else:
			assert False

	def jenkins(self, key):
		key += (key << 12)
		key ^= (key >> 22)
		key += (key << 4)
		key ^= (key >> 9)
		key += (key << 10)
		key ^= (key >> 2)
		key += (key << 7)
		key ^= (key >> 12)
		return key;


	def folded_xor(self, value, num_folds):
		mask = 0 
		bits_in_fold = 64//num_folds
		if num_folds==2:
			mask = 0xffffffff
		else:
			mask = (1 << bits_in_fold) -1

		folded_value = 0
		for fold in range (num_folds):
			#print(folded_value, value, fold, bits_in_fold, mask)
			folded_value = folded_value ^ ((value >> (fold * bits_in_fold)) & mask)

		return folded_value

	def getMaxAction(self, state):
		max_q_value = q_value = 0.0 
		selected_action = init_index = 0

		if not le_featurewise_enable_action_fallback:
			max_q_value = self.retrieveQ(state, 0)
			init_index = 1

		for action in range(init_index, self.m_actions):
			q_value = self.retrieveQ(state, action)
			if q_value > max_q_value:
				max_q_value = q_value
				selected_action = action

		return selected_action

	def updateQ(self, state1, action1, reward, state2, action2):
		tile_index1 = tile_index2 = 0
		QSa1_old_overall = self.retrieveQ(state1, action1)
		QSa2_old_overall = self.retrieveQ(state2, action2)

		for tiling in range (self.m_num_tilings):
			tile_index1 = self.get_tile_index(tiling, state1)
			tile_index2 = self.get_tile_index(tiling, state2)
			Qsa1 = self.m_qtable[tiling][tile_index1][action1]
			Qsa2 = self.m_qtable[tiling][tile_index2][action2]
			Qsa1_old = Qsa1 
			#SARSA
			Qsa1 = Qsa1 + self.m_alpha * (reward + self.m_gamma * Qsa2 - Qsa1)
			self.m_qtable[tiling][tile_index1][action1] = Qsa1




class LearningEngineFeaturewise:
	def __init__(self, alpha, gamma, epsilon, actions, seed, policy, type, zero_init):
		self.type = None
		self.m_alpha = alpha
		self.m_gamma = gamma
		self.m_epsilon = epsilon
		self.m_actions = actions
		self.m_seed = seed

		self.m_max_q_value = 0
		self.m_max_q_value_buckets = 0
		self.m_max_q_value_histogram = 0

		self.m_feature_knowledges = [None] * FeatureType['NumFeatureTypes']

		for index in range(len(le_featurewise_active_features)):
			self.m_feature_knowledges[le_featurewise_active_features[index]] = FeatureKnowledge(le_featurewise_active_features[index],
				alpha, gamma, actions, le_featurewise_feature_weights[index], 
				le_featurewise_weight_gradient, le_featurewise_num_tilings[index],
				le_featurewise_num_tiles[index], zero_init, le_featurewise_hash_types[index],
				le_featurewise_enable_tiling_offset[index])

		m_max_q_value = (1/(1-gamma)) * sum(le_featurewise_num_tilings)


		self.Policy = {'InvalidPolicy' : 0, 'EGreedy' : 1, 'NumPolicies' : 2}
		LearningType = {'InvalidLearningType' : 0, 'QLearning': 1, 'SARSA' : 2, 'NumLearningTypes': 3}
		

		self.stats = { 'action' : {'called' : 0, 'explore' : 0, 'exploit' : 0,
		'dist' : [[0]*2]*MAX_ACTIONS, 'fallback' : 0, 'dyn_fallback_saved_bw' : 0,
		'dyn_fallback_saved_bw_acc' : 0},

		'learn' : {'called' : 0, 'su_skip' : [0]*FeatureType['NumFeatureTypes']},

		'consensus' : {'total' : 0, 'feature_align_dist' : [0]*FeatureType['NumFeatureTypes'],
		'feature_align_all' : 0}
		}

	def chooseAction(self, state, max_to_avg_q_ratio, consensus_vec):
		self.stats['action']['called'] += 1
		action = 0
		max_to_avg_q_ratio = 0.0
		consensus_vec = [False] * FeatureType['NumFeatureTypes']

		if random.choices([True, False], [self.m_epsilon, 1-self.m_epsilon])[0]:
			action = random.randint(0,MAX_ACTIONS-1)
			self.stats['action']['explore'] += 1
			self.stats['action']['dist'][action][0] += 1

		else:
			ret = self.getMaxAction(state, 0.0, max_to_avg_q_ratio, consensus_vec)
			action, max_q, max_to_avg_q_ratio, consensus_vec = ret[0], ret[1], ret[2], ret[3]
			self.stats['action']['exploit'] += 1
			self.stats['action']['dist'][action][1] += 1

		return [action, max_to_avg_q_ratio, consensus_vec] 


	def getMaxAction(self, state, max_q, max_to_avg_q_ratio, consensus_vec):
		max_q_value = q_value = total_q_value = 0.0
		selected_action = init_index = 0

		fallback = self.do_fallback(state)

		if not fallback:
			max_q_value = self.consultQ(state, 0)
			total_q_value += max_q_value
			init_index = 1

		for action in range (init_index, self.m_actions):
			q_value = self.consultQ(state, 0)
			total_q_value += q_value
			if q_value > max_q_value:
				max_q_value = q_value
				selected_action = action 

		if fallback and max_q_value == 0.0:
			self.stats['action']['fallback'] += 1

		# max to avg ratio calculation
		avg_q_value = total_q_value / self.m_actions
		if (max_q_value > 0 and avg_q_value > 0) or (max_q_value < 0 and avg_q_value < 0):
			max_to_avg_q_ratio = abs(max_q_value)//abs(avg_q_value) - 1
		else:
			max_to_avg_q_ratio = (max_q_value - avg_q_value)//abs(avg_q_value)

		if max_q_value < le_featurewise_max_q_thresh * self.m_max_q_value:
			max_to_avg_q_ratio = 0.0 
		max_q = max_q_value

		consensus_vec = self.action_selection_consensus(state, selected_action, consensus_vec)
		return [selected_action, max_q, max_to_avg_q_ratio, consensus_vec]

	# consensus stats: whether each feature's maxAction decision aligns with the final selected action 
	def action_selection_consensus(self, state, selected_action, consensus_vec):
		self.stats['consensus']['total'] += 1
		all_features_align = True 
		for index in range (FeatureType['NumFeatureTypes']):
			if self.m_feature_knowledges[index]:
				if self.m_feature_knowledges[index].getMaxAction(state) == selected_action:
					self.stats['consensus']['feature_align_dist'][index] += 1
					consensus_vec[index] = True 
				else:
					all_features_align = False

		if all_features_align:
			self.stats['consensus']['feature_align_all'] += 1

		return consensus_vec

	def consultQ(self, state, action):
		assert action < self.m_actions
		q_value = 0.0
		maxm = -1000000000.0

		for index in range (FeatureType['NumFeatureTypes']):
			if self.m_feature_knowledges[index]:
				if le_featurewise_pooling_type == 1: # sum pooling
					q_value+= self.m_feature_knowledges[index].retrieveQ(state, action)
				elif le_featurewise_pooling_type == 2: # max pooling
					tmp = self.m_feature_knowledges[index].retrieveQ(state, action)
					if tmp>= maxm:
						maxm = tmp
						q_value = tmp
				else:
					assert False

		return q_value



	def do_fallback(self, state):
		if not le_featurewise_enable_dyn_action_fallback:
			return le_featurewise_enable_action_fallback

		if state.is_high_bw:
			self.stats['action']['dyn_fallback_saved_bw'] += 1
			return False 

		elif state.bw_level >= le_featurewise_bw_acc_check_level and state.acc_level <= le_featurewise_acc_thresh:
			self.stats['action']['dyn_fallback_saved_bw_acc'] += 1
			return False

		return True


	def learn(self, state1, action1, reward, state2, action2, consensus_vec, reward_type):
		self.stats['learn']['called'] += 1
		for index in range(FeatureType['NumFeatureTypes']):
			if self.m_feature_knowledges[index]:
				if (not le_featurewise_selective_update) or consensus_vec[index]:
					self.m_feature_knowledges[index].updateQ(state1, action1, reward, state2, action2)
				elif le_featurewise_selective_update and (not consensus_vec[index]):
					self.stats['learn']['su_skip'][index] += 1
