#ifndef BAIDU_NLP_DU_SENTI_CLASSIFY_FLUID_INCLUDE_WORDSEG_H
#define BAIDU_NLP_DU_SENTI_CLASSIFY_FLUID_INCLUDE_WORDSEG_H

#include <vector>
#include <string>
#include "ilac.h"

namespace senti_cls_dnn { 
    // Max number of tokens
    const int SEN_MAX_TOKENS = 1024;

    class WordSegment {
        /*
         * @brief: WordSegment Tool
         */
        public:
            WordSegment() : _lac_buff(NULL), _results(NULL) {};
            ~WordSegment() {}
            /*
             * @brief：Global Resources Init
             * @param<int>：config_path，config path
             * @return：0,success; -1,failed
             */
            static int global_init(const std::string& config_path);
            /*
             * @brief：Global Resources Destroy
             * @return：NULL
             */
            static void global_destroy();
            /*
             * @brief：Thread Resources Init
             * @return：0,success; -1,failed
             */
            int thread_init();
            /*
             * @brief：Thread Resources Destroy
             * @return：NULL
             */
            void thread_destroy();
            /*
             * @brief：Function for WordSegment
             * @param<int>：str_input，the input string
             * @param<out>：word_list，result vector
             * @return：0,success; -1,failed
             */
            int word_segment(const std::string& str_input,
                std::vector<std::string>& word_list);
        private:
            // thread resources for lac
            void* _lac_buff;
            // return result of lac
            tag_t* _results;
            // global resources for lac
            static void* _s_lac_handle;
    };
}
#endif
