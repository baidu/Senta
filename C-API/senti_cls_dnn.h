#ifndef BAIDU_NLP_DU_SENTI_CLASSIFY_FLUID_INCLUDE_SENTI_CLS_DNN_H
#define BAIDU_NLP_DU_SENTI_CLASSIFY_FLUID_INCLUDE_SENTI_CLS_DNN_H
#include <string>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/io.h"
#include "wordseg.h"

namespace senti_cls_dnn {

class SentiClsRes {
    /*
     * @brief: Sentiment Classification Result
     */
    public:
        SentiClsRes() : _label(1), _pos_prob(0.5),
            _neg_prob(0.5), _confidence_val(1.0) {}
    public:
        // Sentiment Classification Label
        // (0:negative, 1:neural, 2:positive)
        int _label;
        // Positive probability
        float _pos_prob;
        // Negative probability
        float _neg_prob;
        // Confidence score
        float _confidence_val;
};

class SentiClsDnn {
    /*
     * @brief: Sentiment Classification Tool
     */
    public:
        /*
         * @brief: Global Resources Init
         * @param<in>: config_path, the config path
         * @return：0,success; -1,failed
         */
        static int global_init(const std::string& config_path);
        /*
         * @brief: Global Resources Destroy
         * @return: NULL
         */
        static void global_destroy();
        /*
         * @brief: Thread Resources Init
         * @param<in>: thread_id, current thread id
         * @return：0,success; -1,failed
         */
        int thread_init(int thread_id);
        /*
         * @brief: Thread Resources Destroy
         * @return: NULL
         */
        void thread_destroy();
        /*
         * @brief: Predict Function
         * @param<in>：input_str, the input str(encoding:utf8)
         * @param<out>：senti_cls_res, Sentiment Classification Result
         */
        int predict(const std::string& input_str, SentiClsRes& senti_cls_res);
    private:
        /*
         * @brief：Vocab dict init
         * @param<int>：dict_path, the vocab_dict path
         * @return：0,success; -1:failed
         */
        static int init_vocab_dict(const std::string& dict_path);
        /*
         * @brief: Word2Id
         * @param<int>：word_list, words vector
         * @param<out>：id_list，ids vector
         * @return：0,success; -1,failed
         */
        int trans_word_to_id(const std::vector<std::string>& word_list,
            std::vector<int64_t>& id_list);
        /*
         * @brief: Compute the sentiment classification label 
         *      and confidence score
         * @param<out>：senti_cls_res，Sentiment Classification Result
         * @return: NULL
         */
        void normalize_result(SentiClsRes& senti_cls_res);
    private:
        // place variable for fluid inference
        static paddle::platform::CPUPlace _s_place;
        // scope variable for fluid inference
        static paddle::framework::Scope* _s_scope_ptr;
        // executor variable for fluid inference
        static paddle::framework::Executor* _s_executor_ptr;
        // inference_pargram variable for fluid inference
        static std::unique_ptr<paddle::framework::ProgramDesc> 
            _s_inference_program;
        // the map for word2id
        static std::map<std::string, int64_t> _s_vocab_dict;
        // copy_program for fluid inference(used for multi-thread)
        std::unique_ptr<paddle::framework::ProgramDesc> _copy_program;
        // Word Segment tool(depend on the lac)
        WordSegment _wordseg_tool;
        // current thread id
        int _thread_id;
        // the feed_holder_name for current thread
        std::string _feed_holder_name;
        // the fetch_holder_name for current thread
        std::string _fetch_holder_name;
};
}

#endif
