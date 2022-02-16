import learning_engine_featurewise
from collections import deque

#---- /config/pythia.ini----#

# scooby related knobs
scooby_alpha = 0.006508802942367162
scooby_gamma = 0.556300959940946
scooby_epsilon = 0.0018228444309622588
scooby_state_num_bits = 10
scooby_seed = 200
scooby_policy = "EGreedy"
scooby_learning_type = "SARSA"
scooby_actions = [1,3,4,5,10,11,12,22,23,30,32,-1,-3,-6,0]
scooby_max_actions = 64
scooby_pt_size = 256
scooby_st_size = 64
scooby_max_pcs = 5
scooby_max_offsets = 5
scooby_max_deltas = 5
scooby_reward_correct_timely = 20
scooby_reward_correct_untimely = 12
scooby_reward_incorrect = -8
scooby_reward_none = -4
scooby_brain_zero_init = False
scooby_enable_reward_all = False
scooby_enable_track_multiple = False
scooby_enable_reward_out_of_bounds = True
scooby_reward_out_of_bounds = -12
scooby_state_type = 1
scooby_state_hash_type = 11
scooby_access_debug = False
scooby_print_access_debug = False
scooby_enable_state_action_stats = True
scooby_enable_reward_tracker_hit = False
scooby_reward_tracker_hit = -2
scooby_pref_degree = 1
scooby_enable_dyn_degree = False#True
scooby_max_to_avg_q_thresholds = [0.5,1,2]
scooby_dyn_degrees = [1,2,4,4]
scooby_early_exploration_window = 0
scooby_multi_deg_select_type = 2
scooby_last_pref_offset_conf_thresholds = [1,3,8]
scooby_dyn_degrees_type2 = [1,2,4,6]
scooby_action_tracker_size = 2
scooby_enable_hbw_reward = True
scooby_reward_hbw_none = -2
scooby_reward_hbw_incorrect = -14
scooby_reward_hbw_correct_untimely = 12
scooby_reward_hbw_correct_timely = 20
scooby_reward_hbw_out_of_bounds = -12
scooby_reward_hbw_tracker_hit = -2
scooby_last_pref_offset_conf_thresholds_hbw = [1,3,8]
scooby_dyn_degrees_type2_hbw = [1,2,4,6]

# Learning engines
scooby_enable_featurewise_engine = True

# Engine knobs
le_featurewise_active_features = [0,10]
le_featurewise_num_tilings = [3,3]
le_featurewise_num_tiles = [128,128]
le_featurewise_hash_types = [2,2]
le_featurewise_enable_tiling_offset = [1,1]
le_featurewise_max_q_thresh = 0.50
le_featurewise_enable_action_fallback = True
le_featurewise_feature_weights=[1.00,1.00]
le_featurewise_enable_dynamic_weight = False
le_featurewise_weight_gradient = 0.001
le_featurewise_disable_adjust_weight_all_features_align = True
le_featurewise_selective_update = False
le_featurewise_pooling_type = 2
le_featurewise_enable_dyn_action_fallback = True
le_featurewise_bw_acc_check_level = 1
le_featurewise_acc_thresh = 2

#---#

#--- /src/knobs.cc ---#

scooby_high_bw_thresh = 4
scooby_max_states = 1024

#----#

LOG2_PAGE_SIZE = 12
LOG2_BLOCK_SIZE = 6

#---- /scooby_helper.cc ---#

DELTA_SIG_MAX_BITS = 12
DELTA_SIG_SHIFT = 3
PC_SIG_MAX_BITS = 32
PC_SIG_SHIFT = 4
OFFSET_SIG_MAX_BITS = 24
OFFSET_SIG_SHIFT = 4

SIG_SHIFT = 3
SIG_BIT = 12
SIG_MASK = ((1 << SIG_BIT) - 1)
SIG_DELTA_BIT = 7

MAX_REWARDS = 16

#----#

RewardType = {'none' : 0, 'incorrect' : 1, 'correct_untimely' : 2, 'correct_timely' : 3, 'out_of_bounds' : 4, 'tracker_hit' : 5, 'num_rewards' : 6}


class Prefetcher:
	def __init__(self):
		self.type = 0 # single underscore for protected member

	def get_type():
		return self.type 

class Scooby(Prefetcher): # child of class prefetcher
	def __init__(self):
		self.signature_table = deque([])
		self.prefetch_tracker = deque([])
		self.last_evicted_tracker = None
		self.bw_level = 0
		self.core_ipc = 0
		self.acc_level = 0

		self.recorder = 0
		self.target_action_state = {}

		self.state_action_dist = {}
		self.state_action_dist2 = {}
		self.action_deg_dist = {}

		self.Actions = [0]*scooby_max_actions
		for i in range(len(scooby_actions)):
			self.Actions[i] = scooby_actions[i]

		self.brain_featurewise = learning_engine_featurewise.LearningEngineFeaturewise(scooby_alpha,
			scooby_gamma, scooby_epsilon, scooby_max_actions, scooby_seed, scooby_policy,
			scooby_learning_type, scooby_brain_zero_init)


		self.stats = { 'st' : {'lookup' : 0, 'hit' : 0, 'evict' : 0, 'insert' : 0, 'streaming' : 0},

		'predict' : {'called' : 0, 'out_of_bounds' : 0, 'action_dist' : [0]*scooby_max_actions, 'issue_dist' : [0]*scooby_max_actions,
		'pred_hit' : [0]*scooby_max_actions, 'out_of_bounds_dist' : [0]*scooby_max_actions, 'predicted' : 0, 'multi_deg' : 0, 
		'multi_deg_called' : 0, 'multi_deg_histogram' : [0]*(16+1), 'deg_histogram' : [0]*(16+1) }, #define MAX_SCOOBY_DEGREE 16

		'track' : {'called' : 0, 'same_address' : 0, 'evict' : 0},

		'reward' : { 'demand' : {'called' : 0, 'pt_not_found' : 0, 'pt_found' : 0,
								'pt_found_total' : 0, 'has_reward' : 0},

					  'train' : {'called' : 0},

					  'assign_reward' : {'called' : 0},

					  'compute_reward' : {'dist' : [[0]*2]*MAX_REWARDS},

					  'correct_timely' : 0, 'correct_untimely' : 0, 'no_pref' : 0, 
					  'incorrect' : 0, 'out_of_bounds' : 0, 'tracker_hit' : 0, 
					  'dist' : [[0] * (16)] * (64) # define MAX_ACTIONS 64 define MAX_REWARDS 16

					},

		'train' : {'called' : 0, 'compute_reward' : 0},

		'register_fill' : {'called' : 0, 'set' : 0, 'set_total' : 0},

		'register_prefetch_hit' : {'called' : 0, 'set' : 0, 'set_total' : 0},

		'pref_issue' : {'scooby' : 0},

		'bandwidth' : {'epochs' : 0, 'histogram' : [0]*4}, #define DRAM_BW_LEVELS 4

		'ipc' : {'epochs' : 0, 'histogram' : [0]*4}, #define SCOOBY_MAX_IPC_LEVEL 4

		'cache_acc' : {'epochs' : 0, 'histogram' : [0]*10} #define CACHE_ACC_LEVELS 10

		}

		self.recorder = { 'unique_pcs' : set(), 'unique_trigger_pcs' : set(), 'unique_pages' : set(), 
		'access_bitmap_dist' : {}, 'hop_delta_dist' : [[0] * 127] * (16+1),  #define MAX_HOP_COUNT 16
		'total_bitmaps_seen' : 0, 'unique_bitmaps_seen' : 0, 'pc_bw_dist' : {}
		}

	def recorder_record_access (self, pc, address, page, offset, bw_level):
		self.recorder['unique_pcs'].add(pc)
		self.recorder['unique_pages'].add(page)


	# Private members (double underscore)

	def invoke_prefetcher(self, pc, address, cache_hit, type, pref_addr):
		page = address >> LOG2_PAGE_SIZE
		offset = (address >> LOG2_BLOCK_SIZE) & ((1 << (LOG2_PAGE_SIZE - LOG2_BLOCK_SIZE)) - 1)
		self.reward(address)

		self.recorder_record_access(pc, address, page, offset, self.bw_level)

		#self.update_global_state(pc, page, offset, address)

		stentry = self.update_local_state(pc, page, offset, address)
 
		state = State()
		state.pc = pc 
		state.address = address 
		state.page = page 
		state.offset = offset 
		state.delta = 0
		if stentry.deltas:
			state.delta = stentry.deltas[-1]
		state.local_delta_sig = stentry.get_delta_sig()
		state.local_delta_sig2 = stentry.get_delta_sig2()
		state.local_pc_sig = stentry.get_pc_sig()
		state.local_offset_sig = stentry.get_offset_sig()
		state.bw_level = self.bw_level
		state.is_high_bw = self.is_high_bw()
		state.acc_level = self.acc_level

		count = len(pref_addr)
		self.predict(address, page, offset, state, pref_addr)
		self.stats['pref_issue']['scooby'] += (len(pref_addr) - count)


	def predict(self, base_address, page, offset, state, pref_addr):
		self.stats['predict']['called'] += 1
		action_index = 0
		pref_degree = scooby_pref_degree
		consensus_vec = []

		if scooby_enable_featurewise_engine:
			ret = self.brain_featurewise.chooseAction(state, 1.0, [])
			action_index, max_to_avg_q_ratio, consensus_vec = ret[0], ret[1], ret[2]
			if scooby_enable_dyn_degree:
				pref_degree = self.get_dyn_pref_degree(max_to_avg_q_ratio, page, self.Actions[action_index])

		

		addr = 0xdeadbeef
		ptentry = None 
		predicted_offset = 0 
		if self.Actions[action_index] != 0:
			predicted_offset = offset + self.Actions[action_index]
			if predicted_offset >=0 and predicted_offset < 64:
				addr = (page << LOG2_PAGE_SIZE) + (predicted_offset << LOG2_BLOCK_SIZE)
				ret = self.track(addr, state, action_index, ptentry)
				new_addr, ptentry = ret[0], ret[1]

				if new_addr:
					pref_addr.append(addr)
					self.track_in_st(page, predicted_offset, self.Actions[action_index])
					self.stats['predict']['issue_dist'][action_index] += 1
					if pref_degree > 1:
						self.gen_multi_degree_pref(page, offset, self.Actions[action_index], pref_degree, pref_addr)
					self.stats['predict']['deg_histogram'][pref_degree] += 1
					ptentry.consensus_vec = consensus_vec

				else:
					self.stats['predict']['pred_hit'][action_index] += 1
					if scooby_enable_reward_tracker_hit:
						addr = 0xdeadbeef
						ret = self.track(addr, state, action_index, ptentry)
						ptentry = ret[1]
						assert ptentry
						self.assign_reward(ptentry, RewardType['out_of_bounds'])
						ptentry.consensus_vec = consensus_vec

		else:
			addr = 0xdeadbeef
			ptentry = self.track(addr, state, action_index, ptentry)[1]
			self.stats['predict']['action_dist'][action_index] += 1
			ptentry.consensus_vec = consensus_vec

		self.stats['predict']['predicted'] += len(pref_addr)
		return len(pref_addr)


	def gen_multi_degree_pref(self, page, offset, action, pref_degree, pref_addr):
		self.stats['predict']['multi_deg_called'] += 1
		addr = 0xdeadbeef
		predicted_offset = 0
		if action != 0:
			for degree in range (2, pref_degree+1):
				predicted_offset = offset + degree * action
				if predicted_offset >= 0 and predicted_offset < 64:
					addr = (page << LOG2_PAGE_SIZE) + (predicted_offset << LOG2_BLOCK_SIZE)
					pref_addr.append(addr)
					self.stats['predict']['multi_deg'] += 1
					self.stats['predict']['multi_deg_histogram'][degree] += 1
	

	def track_in_st(self, page, pred_offset, pref_offset):
		st_index = -1
		for i in range (len(self.signature_table)):
			if self.signature_table[i].page == page:
				st_index = i
				break

		if st_index!=-1:
			self.signature_table[st_index].track_prefetch(pred_offset, pref_offset)




	def track(self, address, state, action_index, tracker):
		self.stats['track']['called'] += 1
		new_addr = True 
		ptentries = self.search_pt(address, False)
		if len(ptentries) == 0:
			new_addr = True 
		else:
			new_addr = False

		if not new_addr and address != 0xdeadbeef and not scooby_enable_track_multiple:
			self.stats['track']['same_address'] += 1
			tracker = None 
			return [new_addr, tracker]


		# new prefetched address that hasn't been seen before
		ptentry = None 
		if len(self.prefetch_tracker) >= scooby_pt_size:
			self.stats['track']['evict'] += 1
			ptentry = self.prefetch_tracker[0]
			self.prefetch_tracker.popleft()
			if self.last_evicted_tracker:
				self.train(ptentry, self.last_evicted_tracker)
			self.last_evicted_tracker = ptentry

		ptentry = Scooby_PTEntry(address, state, action_index)
		self.prefetch_tracker.append(ptentry)
		assert len(self.prefetch_tracker) <= scooby_pt_size

		return [new_addr, ptentry]


	def train(self, curr_evicted, last_evicted):
		self.stats['train']['called'] += 1
		if not last_evicted.has_reward:
			self.stats['train']['compute_reward'] += 1
			last_evicted = self.reward_pt(last_evicted)

		assert last_evicted.has_reward

		if scooby_enable_featurewise_engine:
			self.brain_featurewise.learn(last_evicted.state, last_evicted.action_index,
				last_evicted.reward, curr_evicted.state, curr_evicted.action_index,
				last_evicted.consensus_vec, last_evicted.reward_type)





	def get_dyn_pref_degree(self, max_to_avg_q_ratio, page, action):
		counted = False 
		degree = 1

		if scooby_multi_deg_select_type == 2:
			st_index = -1
			for i in range (len(self.signature_table)):
				if self.signature_table[i].page == page:
					st_index = i 
					break

			if st_index!=-1:
				conf = 0
				ret = self.signature_table[st_index].search_action_tracker(action, conf)
				found, conf = ret[0], ret[1]

				conf_thresholds = deg_normal = []
				if self.is_high_bw():
					conf_thresholds = scooby_last_pref_offset_conf_thresholds_hbw
					deg_normal = scooby_dyn_degrees_type2_hbw
				else:
					conf_thresholds = scooby_last_pref_offset_conf_thresholds
					deg_normal = scooby_dyn_degrees_type2

				if found:
					for index in range(len(conf_thresholds)):
						if conf <= conf_thresholds[index]:
							degree = deg_normal[index]
							counted = True 
							break

					if not counted:
						degree = deg_normal[-1]

				else:
					degree = 1

		return degree


	def update_local_state(self, pc, page, offset, address):
		self.stats['st']['lookup'] += 1

		stentry = None
		found = -1
		for i in range (len(self.signature_table)):
			if self.signature_table[i].page == page:
				found = i 
				break

		if found != -1:
			self.stats['st']['hit'] += 1
			stentry = self.signature_table[i] 
			stentry.update(page, pc, offset, address)
			self.signature_table.remove(self.signature_table[i])
			self.signature_table.append(stentry)
			return stentry 

		else:
			if len(self.signature_table) >= scooby_st_size:
				self.stats['st']['evict'] += 1
				stentry = self.signature_table[0]
				self.signature_table.popleft()

			self.stats['st']['insert'] += 1
			stentry = Scooby_STEntry(page, pc, offset)
			self.signature_table.append(stentry)
			return stentry



	def reward(self, address):
		ptentries = self.search_pt(address, scooby_enable_reward_all) # rewarding all entries of that address

		if len(ptentries) == 0:
			self.stats['reward']['demand']['pt_not_found'] += 1
			return

		else:
			self.stats['reward']['demand']['pt_found'] += 1

		size = len(ptentries)

		for index in range(size):
			ptentry = ptentries[index]
			self.stats['reward']['demand']['pt_found_total'] += 1

			if ptentry.has_reward:
				self.stats['reward']['demand']['has_reward'] += 1
				return

			if ptentry.is_filled:
				self.assign_reward(ptentry, RewardType['correct_timely'])

			else:
				self.assign_reward(ptentry, RewardType['correct_untimely'])

			ptentry.has_reward = True

	def reward_pt(self, ptentry):
		self.stats['reward']['train']['called'] += 1
		assert not ptentry.has_reward

		if ptentry.address == 0xdeadbeef: # no prefetch
			self.assign_reward(ptentry, RewardType['none'])
		else: # incorrect prefetch
			self.assign_reward(ptentry, RewardType['incorrect'])

		ptentry.has_reward = True
		return ptentry


	def assign_reward(self, ptentry, type):
		assert not ptentry.has_reward

		reward = self.compute_reward(ptentry, type)

		ptentry.reward = reward
		ptentry.reward_type = type
		ptentry.has_reward = True 

		self.stats['reward']['assign_reward']['called'] += 1

		if type == RewardType['correct_timely']:
			self.stats['reward']['correct_timely'] += 1

		elif type == RewardType['correct_untimely']:
			self.stats['reward']['correct_untimely'] += 1

		elif type == RewardType['incorrect']:
			self.stats['reward']['incorrect'] += 1

		elif type == RewardType['none']:
			self.stats['reward']['no_pref'] += 1

		elif type == RewardType['out_of_bounds']:
			self.stats['reward']['out_of_bounds'] += 1

		elif type == RewardType['tracker_hit']:
			self.stats['reward']['tracker_hit'] += 1

		else:
			assert False

		self.stats['reward']['dist'][ptentry.action_index][type] += 1


	def compute_reward(self, ptentry, type):
		high_bw = None 
		if scooby_enable_hbw_reward and self.is_high_bw():
			high_bw = True 
		else:
			high_bw = False

		reward = 0
		self.stats['reward']['compute_reward']['dist'][type][high_bw] += 1

		if type == RewardType['correct_timely']:
			if high_bw:
				reward = scooby_reward_hbw_correct_timely
			else:
				reward = scooby_reward_correct_timely

		elif type == RewardType['correct_untimely']:
			if high_bw:
				reward = scooby_reward_hbw_correct_untimely
			else:
				reward = scooby_reward_correct_untimely

		elif type == RewardType['incorrect']:
			if high_bw:
				reward = scooby_reward_hbw_incorrect
			else:
				reward = scooby_reward_incorrect

		elif type == RewardType['none']:
			if high_bw:
				reward = scooby_reward_hbw_none
			else:
				reward = scooby_reward_none

		elif type == RewardType['out_of_bounds']:
			if high_bw:
				reward = scooby_reward_hbw_out_of_bounds
			else:
				reward = scooby_reward_out_of_bounds

		elif type == RewardType['tracker_hit']:
			if high_bw:
				reward = scooby_reward_hbw_tracker_hit
			else:
				reward = scooby_reward_tracker_hit

		else:
			print("Invalid reward type found", end = " ")
			print(type + '\n')
			assert False

		return reward


	def is_high_bw(self):
		if self.bw_level >= scooby_high_bw_thresh:
			return True 
		else: return False



	def search_pt(self, address, search_all):
		entries = []
		size = len(self.prefetch_tracker)

		for index in range (size):
			if self.prefetch_tracker[index].address == address:
				entries.append(self.prefetch_tracker[index])
				if not search_all:
					break

		return entries



	# Public members (no underscore)



class Scooby_PTEntry:
	def __init__ (self, ad, st, ac):
		self.address = ad
		self.state = st
		self.action_index = ac
		self.is_filled = False # set when prefetched line is filled into cache check during reward to measure timeliness
		self.pf_cache_hit = False # set when prefetched line is alredy found in cache
		self.reward = 0
		self.reward_type = RewardType['none']
		self.has_reward = False

class ActionTracker:
	def __init__(self, act, c):
		self.action = act
		self.conf = c


class Scooby_STEntry:
	def __init__(self, page, pc, offset):
		self.page = 0
		self.pcs = deque([pc])
		self.offsets = deque([offset])
		self.deltas = deque([])
		self.bmp_real = [0]*64
		self.bmp_pred = [0]*64
		self.unique_pcs = set([pc])
		self.unique_deltas = set()
		self.trigger_pc = pc
		self.trigger_offset = offset
		self.streaming = False 

		self.action_tracker = deque([])
		self.action_with_max_degree = set()
		self.afterburning_actions = set()

		self.total_prefetches = 0

		self.unique_pcs.add(pc)
		self.bmp_real[offset] = 1

	def update(self, page, pc, offset, address):
		assert self.page == page 

		#insert PC
		if len(self.pcs) >= scooby_max_pcs:
			self.pcs.popleft()
		self.pcs.append(pc)
		self.unique_pcs.add(pc)

		#insert deltas
		if self.offsets:
			self.delta = None 
			if offset>self.offsets[-1]: self.delta = offset - self.offsets[-1]
			else: self.delta = (-1) * (self.offsets[-1] - offset)

			if len(self.deltas) >= scooby_max_deltas:
				self.deltas.popleft()
			self.deltas.append(self.delta)
			self.unique_deltas.add(self.delta)

		#insert offset
		if len(self.offsets) >= scooby_max_offsets:
			self.offsets.popleft()
		self.offsets.append(offset)

		self.bmp_real[offset] = 1


	def get_delta_sig(self):
		signature = 0
		delta = 0

		n = len(self.deltas)
		ptr = 0
		if n >= 4:
			ptr = n - 4

		for index in range (ptr, n):
			signature = signature << DELTA_SIG_SHIFT
			signature = signature & ((1<<DELTA_SIG_MAX_BITS)-1)
			delta = self.deltas[index] & ((1<<7)-1)
			signature = signature ^ self.delta
			signature = signature & ((1<<DELTA_SIG_MAX_BITS)-1)

		return signature

	def get_delta_sig2(self):
		curr_sig = 0
		n = len(self.deltas)
		ptr = 0
		if n >= 4:
			ptr = n - 4

		for index in range (ptr, n):
			sig_delta = self.deltas[index]
			if self.deltas[index] < 0:
				sig_delta = ((-1 * self.deltas[index]) + (1 << (SIG_DELTA_BIT - 1)))
			curr_sig = ((curr_sig<<SIG_SHIFT) ^ sig_delta) & SIG_MASK

		return curr_sig


	def get_pc_sig(self):
		signature = 0 
		n = len(self.pcs)
		ptr = 0
		if n >= 4:
			ptr = n-4

		for index in range(ptr, n):
			signature = signature << PC_SIG_SHIFT
			signature = signature ^ self.pcs[index]

		signature = signature & ((1 << PC_SIG_MAX_BITS) - 1)
		return signature 

	def get_offset_sig(self):
		signature = 0 
		n = len(self.offsets)
		ptr = 0
		if n >= 4:
			ptr = n-4

		for index in range(ptr, n):
			signature = signature << OFFSET_SIG_SHIFT
			signature = signature ^ self.offsets[index]

		signature = signature & ((1<<OFFSET_SIG_MAX_BITS) - 1)
		return signature

	def search_action_tracker(self, action, conf):
		conf = 0
		it = -1
		for i in range (len(self.action_tracker)):
			if self.action_tracker[i].action == action:
				it = i 
				break

		if it == -1:
			return [False, conf]
		else:
			return [True, self.action_tracker[it].conf]

	def track_prefetch(self, pred_offset, pref_offset):
		if self.bmp_pred[pred_offset] == 0:
			self.bmp_pred[pred_offset] = 1
			self.total_prefetches += 1

			self.insert_action_tracker(pref_offset)


	def insert_action_tracker(self, pref_offset):
		it = -1
		for i in range(len(self.action_tracker)):
			if self.action_tracker[it].action == pref_offset:
				it = i
				break

		tmp = self.action_tracker[it]

		if it != -1:
			self.action_tracker[it].conf += 1
			self.action_tracker.remove(tmp)
			self.action_tracker.append(tmp)

		else:
			if len(self.action_tracker) >= scooby_action_tracker_size:
				self.action_tracker.popleft()
			self.action_tracker.append(ActionTracker(pref_offset, 0))


class State:
	def __init__(self):
		self.pc = 0xdeadbeef
		self.address = 0xdeadbeef
		self.page = 0xdeadbeef
		self.offset = 0
		self.delta = 0
		self.local_delta_sig = 0
		self.local_delta_sig2 = 0
		self.local_pc_sig = 0
		self.local_offset_sig = 0
		self.bw_level = 0
		self.is_high_bw = False 
		self.acc_level = 0

	def value(self):
		value = 0
		if scooby_state_type == 1:
			value = self.pc
		elif scooby_state_type == 2:
			value = self.pc
			value = value << 6
			value += self.offset 
		elif scooby_state_type == 3:
			value = self.offset
		elif scooby_state_type == 4:
			value = self.local_delta_sig
		elif scooby_state_type == 5:
			value = self.local_pc_sig
		elif scooby_state_type == 6:
			value = self.local_delta_sig2
		else:
			assert False 

		hashed_value = self.get_hash(value)
		return hashed_value


	def get_hash(self, key):
		value = 0
		if scooby_state_hash_type == 11:
			value = self.Wang6shift(key)
			value = (value % scooby_max_states)
		else:
			assert False 

		return value


	def Wang6shift(self, key):
		key += ~(key << 15)
		key ^=  (key >> 10)
		key +=  (key << 3)
		key ^=  (key >> 6)
		key += ~(key << 11)
		key ^=  (key >> 16)
		return key;






