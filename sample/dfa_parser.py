import spot

class DFAExtractor:
    def __init__(self, ltlf_formula):
        print(f"[SYSTEM] Compiling LTLf Formula: {ltlf_formula}")
        # 1. 注入有限时序逻辑，生成底层 Büchi 自动机
        f = spot.from_ltlf(ltlf_formula)
        # 2. 翻译并强制生成确定性(det)与基于状态接收(sbacc)的结构
        aut = f.translate('det', 'sbacc')
        # 3. 剥离无限语义，降维成纯正的有限状态自动机 (DFA)
        self.dfa = spot.to_finite(aut)
        # 获取 BDD (二元决策图) 字典，用于将底层的 0/1 矩阵解析回人类可读的字符串
        self.bdict = self.dfa.get_dict()
        
    def get_graph_structure(self):
        dfa_struct = {}
        # 提取初始状态
        init_state = self.dfa.get_init_state_number()
        
        # 遍历所有生成的逻辑状态
        for s in range(self.dfa.num_states()):
            edges = []
            is_accepting = False
            
            # 遍历当前状态射出的所有边
            for t in self.dfa.out(s):
                # 核心：解析出这条边的触发条件 (例如: "ts_1 & !ct_1")
                cond = spot.bdd_format_formula(self.bdict, t.cond)
                if cond == '1': 
                    cond = 'True' # 1 代表无条件驻留
                    
                edges.append({
                    'dst_state': t.dst, 
                    'condition': cond
                })
                
                # 判断当前状态是否满足了最终的 LTLf 目标
                if t.acc:
                    is_accepting = True
                    
            dfa_struct[s] = {
                'edges': edges, 
                'is_accepting': is_accepting
            }
            
        return init_state, dfa_struct

if __name__ == "__main__":
    # --- 逻辑规则定义 ---
    # F 代表 Eventually (最终一定要发生)
    # & 代表 And (并且)
    # 规则：必须先完成个体任务 ts_1，然后在未来的某个时刻，再两机协同完成 ct_1
    formula = "F(ts_1 & F(ct_1))"
    
    extractor = DFAExtractor(formula)
    init_state, graph = extractor.get_graph_structure()
    
    print("\n=== DFA State Machine Extracted into Python Memory ===")
    print(f"[INFO] Initial State ID: {init_state}")
    
    for state_id, info in graph.items():
        # 赛博朋克风高亮输出
        status = "★ [ACCEPTING / GOAL]" if info['is_accepting'] else "  [Intermediate]"
        print(f"\nState {state_id} {status}:")
        
        for edge in info['edges']:
            print(f"  --({edge['condition']})--> Go to State {edge['dst_state']}")