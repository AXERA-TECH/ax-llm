#pragma once
#include "ax_model_runner_ax650.hpp"

#include "utils/axcl_manager.h"
#include "utils/sample_log.h"

#include "untar.h"
#include <memory>

class ax_parallel_runner
{
private:
    std::vector<std::shared_ptr<ax_runner_ax650>> m_runners;
    std::vector<int> m_dev_ids;

    bool b_input_mem_sync = false;
    /* data */

    bool contains(const std::string &str, const std::string &substr)
    {
        return str.find(substr) != std::string::npos;
    }

public:
    int init(std::string model_file, int devid)
    {
        m_dev_ids.clear();
        m_dev_ids.push_back(devid);
        m_runners.clear();
        m_runners.push_back(std::make_shared<ax_runner_ax650>());
        return m_runners[0]->init(model_file.c_str(), devid);
    }

    int init(std::string model_file, std::vector<int> dev_ids)
    {
        const std::string suffix = ".axmodel";
        if (model_file.size() >= suffix.size() &&
            model_file.compare(model_file.size() - suffix.size(), suffix.size(), suffix) == 0)
        {
            return init(model_file, dev_ids[0]);
        }

        this->m_dev_ids = dev_ids;
        std::vector<std::vector<char>> rank_models(dev_ids.size());
        std::vector<std::string> rank_filenames(dev_ids.size());
        std::vector<std::string> rank_names(dev_ids.size());
        for (int i = 0; i < dev_ids.size(); i++)
        {
            rank_names[i] = "_r" + std::to_string(i) + "_";
        }

        untar::tarFile tarFile = untar::tarFile((char *)model_file.c_str(), (untar::tarMode)untar::File);

        auto entries = tarFile.getEntries();
        for (auto &entry : entries)
        {
            std::string filename = entry.first;
            for (int i = 0; i < rank_names.size(); i++)
            {
                if (contains(filename, rank_names[i]))
                {
                    rank_filenames[i] = filename;
                    int filesize;
                    size_t offset;
                    auto fs = entry.second->wantToExtract(&filesize, &offset);
                    rank_models[i].resize(filesize);
                    fs->read((char *)rank_models[i].data(), filesize);
                    break;
                }
            }
        }
        tarFile.close();
        if (rank_models.size() != dev_ids.size())
        {
            ALOGE("model file num %ld not equal to dev_ids num %ld", rank_models.size(), dev_ids.size());
            return -1;
        }
        m_runners.resize(rank_models.size());
        std::vector<int> rets(rank_models.size());
#pragma omp parallel for
        for (int i = 0; i < rank_models.size(); i++)
        {
            if (rank_models[i].size() == 0)
            {
                ALOGE("model file %s not found", rank_filenames[i].c_str());
                rets[i] = -1;
            }
            std::shared_ptr<ax_runner_ax650> runner = std::make_shared<ax_runner_ax650>();
            int ret = runner->init(rank_models[i].data(), rank_models[i].size(), dev_ids[i]);
            if (ret != 0)
            {
                ALOGE("%s init failed, ret=%d", rank_filenames[i].c_str(), ret);
                // return ret;
            }
            else
            {
                ALOGD("%s init success, ret=%d", rank_filenames[i].c_str(), ret);
                m_runners[i] = runner;
            }
            rets[i] = ret;
        }
        for (int i = 0; i < rets.size(); i++)
        {
            if (rets[i] != 0)
            {
                ALOGE("%s init failed, ret=%d", rank_filenames[i].c_str(), rets[i]);
                return rets[i];
            }
        }
        return 0;
    }
    void release()
    {
        deinit();
    }
    void deinit()
    {
        for (auto &runner : m_runners)
        {
            runner->deinit();
        }
        m_runners.clear();
        m_dev_ids.clear();
    }

    std::vector<std::shared_ptr<ax_runner_ax650>> &get_runners() { return m_runners; }

    // Device ID accessors
    std::vector<int> get_devids() { return m_dev_ids; }
    int get_devid() { return get_devid(0); }
    int get_devid(int rankid)
    {
        return (rankid >= 0 && rankid < (int)m_runners.size())
                   ? m_runners[rankid]->get_devid()
                   : -1;
    }

    // Global (rank 0) input accessors
    int get_num_inputs() { return m_runners[0]->get_num_inputs(); }
    int get_num_outputs() { return m_runners[0]->get_num_outputs(); }
    int get_num_groups() { return m_runners[0]->get_num_groups(); }

    const ax_runner_tensor_t &get_input(int idx) { return m_runners[0]->get_input(idx); }
    const ax_runner_tensor_t *get_inputs_ptr() { return m_runners[0]->get_inputs_ptr(); }
    const ax_runner_tensor_t &get_input(const std::string &name) { return m_runners[0]->get_input(name); }
    const ax_runner_tensor_t &get_input(int grpid, int idx) { return m_runners[0]->get_input(grpid, idx); }
    const ax_runner_tensor_t *get_inputs_ptr(int grpid) { return m_runners[0]->get_inputs_ptr(grpid); }
    const ax_runner_tensor_t &get_input(int grpid, const std::string &name) { return m_runners[0]->get_input(grpid, name); }

    /**
     * @brief set input data to all ranks (only host to device)
     *
     * @param data host data
     * @param size size of data
     * @param grpid group id
     * @param idx input id
     * @return int
     */
    int set_input(int grpid, int idx, void *data, int size)
    {
        std::vector<int> rets(m_runners.size());
#pragma omp parallel for
        for (int rankid = 0; rankid < m_runners.size(); rankid++)
        {
            auto input = get_rank_input(rankid, grpid, idx);
            int ret = axcl_Memcpy((void *)input.phyAddr, data, size, AXCL_MEMCPY_HOST_TO_DEVICE, get_devid(rankid));
            rets[rankid] = ret;
        }
        for (int rankid = 0; rankid < m_runners.size(); rankid++)
        {
            if (rets[rankid] != 0)
            {
                ALOGE("axcl_Memcpy failed, ret=%d", rets[rankid]);
                return rets[rankid];
            }
        }
        return 0;
    }

    /**
     * @brief set input data to all ranks group-0 (only host to device)
     *
     * @param data host data
     * @param size size of data
     * @param idx input id
     * @return int
     */
    int set_input(int idx, void *data, int size)
    {
        return set_input(0, idx, data, size);
    }

    int set_input(int grpid, const std::string &name, void *data, int size)
    {
        std::vector<int> rets(m_runners.size());
#pragma omp parallel for
        for (int rankid = 0; rankid < m_runners.size(); rankid++)
        {
            auto input = get_rank_input(rankid, grpid, name);
            int ret = axcl_Memcpy((void *)input.phyAddr, data, size, AXCL_MEMCPY_HOST_TO_DEVICE, get_devid(rankid));
            rets[rankid] = ret;
        }
        for (int rankid = 0; rankid < m_runners.size(); rankid++)
        {
            if (rets[rankid] != 0)
            {
                ALOGE("axcl_Memcpy failed, ret=%d", rets[rankid]);
                return rets[rankid];
            }
        }
        return 0;
    }

    int set_input(const std::string &name, void *data, int size)
    {
        return set_input(0, name, data, size);
    }

    // Global (rank 0) output accessors
    const ax_runner_tensor_t &get_output(int idx) { return m_runners[0]->get_output(idx); }
    const ax_runner_tensor_t *get_outputs_ptr() { return m_runners[0]->get_outputs_ptr(); }
    const ax_runner_tensor_t &get_output(const std::string &name) { return m_runners[0]->get_output(name); }
    const ax_runner_tensor_t &get_output(int grpid, int idx) { return m_runners[0]->get_output(grpid, idx); }
    const ax_runner_tensor_t *get_outputs_ptr(int grpid) { return m_runners[0]->get_outputs_ptr(grpid); }
    const ax_runner_tensor_t &get_output(int grpid, const std::string &name) { return m_runners[0]->get_output(grpid, name); }

    // Rank-specific input accessors
    const ax_runner_tensor_t &get_rank_input(int rankid, int idx)
    {
        return m_runners[rankid]->get_input(idx);
    }
    const ax_runner_tensor_t *get_rank_inputs_ptr(int rankid)
    {
        return m_runners[rankid]->get_inputs_ptr();
    }
    const ax_runner_tensor_t &get_rank_input(int rankid, const std::string &name)
    {
        return m_runners[rankid]->get_input(name);
    }
    const ax_runner_tensor_t &get_rank_input(int rankid, int grpid, int idx)
    {
        return m_runners[rankid]->get_input(grpid, idx);
    }
    const ax_runner_tensor_t *get_rank_inputs_ptr(int rankid, int grpid)
    {
        return m_runners[rankid]->get_inputs_ptr(grpid);
    }
    const ax_runner_tensor_t &get_rank_input(int rankid, int grpid, const std::string &name)
    {
        return m_runners[rankid]->get_input(grpid, name);
    }

    // Rank-specific output accessors
    const ax_runner_tensor_t &get_rank_output(int rankid, int idx)
    {
        return m_runners[rankid]->get_output(idx);
    }
    const ax_runner_tensor_t *get_rank_outputs_ptr(int rankid)
    {
        return m_runners[rankid]->get_outputs_ptr();
    }
    const ax_runner_tensor_t &get_rank_output(int rankid, const std::string &name)
    {
        return m_runners[rankid]->get_output(name);
    }
    const ax_runner_tensor_t &get_rank_output(int rankid, int grpid, int idx)
    {
        return m_runners[rankid]->get_output(grpid, idx);
    }
    const ax_runner_tensor_t *get_rank_outputs_ptr(int rankid, int grpid)
    {
        return m_runners[rankid]->get_outputs_ptr(grpid);
    }
    const ax_runner_tensor_t &get_rank_output(int rankid, int grpid, const std::string &name)
    {
        return m_runners[rankid]->get_output(grpid, name);
    }

    void set_input_mem_sync()
    {
    }

    void input_mem_sync(int grpid)
    {
        // ALOGI("mem_sync");
        for (int i = 0; i < m_runners[0]->get_num_inputs(); i++)
        {
            axcl_Memcpy(m_runners[0]->get_input(grpid, i).pVirAddr,
                        (void *)m_runners[0]->get_input(grpid, i).phyAddr,
                        m_runners[0]->get_input(grpid, i).nSize, AXCL_MEMCPY_DEVICE_TO_HOST, m_runners[0]->get_devid());

            for (int j = 1; j < m_runners.size(); j++)
            {
                axcl_Memcpy((void *)m_runners[j]->get_input(grpid, i).phyAddr,
                            m_runners[0]->get_input(grpid, i).pVirAddr,
                            m_runners[j]->get_input(grpid, i).nSize, AXCL_MEMCPY_HOST_TO_DEVICE, m_runners[j]->get_devid());
            }
        }
    }

    int inference(int grpid)
    {
        if (b_input_mem_sync)
            input_mem_sync(grpid);

        std::vector<int> rets(m_runners.size());
#pragma omp parallel for
        for (int i = 0; i < m_runners.size(); i++)
        {
            int ret = m_runners[i]->inference(grpid);
            rets[i] = ret;
        }
        for (int i = 0; i < m_runners.size(); i++)
        {
            if (rets[i] != 0)
            {
                ALOGE("inference failed, ret=%d", rets[i]);
                return rets[i];
            }
        }

        return 0;
    }

    int inference()
    {
        return inference(0);
    }
};
