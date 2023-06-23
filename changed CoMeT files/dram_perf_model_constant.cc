// ########################################
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>
// ########################################
#include "dram_perf_model_constant.h"
#include "simulator.h"
#include "config.h"
#include "config.hpp"
#include "stats.h"
#include "shmem_perf.h"
#include "dram_trace_collect.h" // Used to calculate the bank number from an address.

#define LOW_POWER 0
#define NORMAL_POWER 1

// ########################################
std::string ACCESS_DATA_FOLDER = "/home/erik/Documents/jaar3-bach/bachelorthesis/code/data/dram_access_data/";
std::string ACCESS_INFO_FILE = "/home/erik/Documents/jaar3-bach/bachelorthesis/code/data/access_info.txt";
std::string ACCESS_DATA_HEADER = "time,core_id,total_accesses,queue_delay,processing_time,m_dram_access_cost,access_latency\n";
std::string ACCESS_DATA_HEADER_RAW = "time,core_id,access_latency\n";
int access_newfile = 0;
int access_interval_start_time = 0;
int total_accesses[MAX_NUM_OF_CORES];
int total_queue_delay[MAX_NUM_OF_CORES];
int total_processing_time[MAX_NUM_OF_CORES];
int total_m_dram_access_cost[MAX_NUM_OF_CORES];
int total_access_latency[MAX_NUM_OF_CORES];
#define ACCUMULATION_TIME     (100) // In nanoseconds
int accumulation = 0;
// ########################################

   // printf("queue_delay(ns):         %lu\n", queue_delay.getNS());
   // printf("processing_time(ns):     %lu\n", processing_time.getNS());
   // printf("m_dram_access_cost(ns):  %lu\n", m_dram_access_cost.getNS());
   // printf("access_latency(ns):      %lu\n", access_latency.getNS());


/*
   This file has been extended to support a low power access latency,
   which will be used when memory DTM is used.
*/

DramPerfModelConstant::DramPerfModelConstant(core_id_t core_id,
      UInt32 cache_block_size):
   DramPerfModel(core_id, cache_block_size),
   m_queue_model(NULL),
   m_dram_bandwidth(8 * Sim()->getCfg()->getFloat("perf_model/dram/per_controller_bandwidth")), // Convert bytes to bits
   m_total_queueing_delay(SubsecondTime::Zero()),
   m_total_access_latency(SubsecondTime::Zero())
{
   m_dram_access_cost = SubsecondTime::FS() * static_cast<uint64_t>(TimeConverter<float>::NStoFS(Sim()->getCfg()->getFloat("perf_model/dram/latency"))); // Operate in fs for higher precision before converting to uint64_t/SubsecondTime

   // Read the low power access cost.
   m_dram_access_cost_lowpower  = SubsecondTime::FS() * static_cast<uint64_t>(TimeConverter<float>::NStoFS(Sim()->getCfg()->getFloat("perf_model/dram/latency_lowpower"))); // Operate in fs for higher precision before converting to uint64_t/SubsecondTime

   if (Sim()->getCfg()->getBool("perf_model/dram/queue_model/enabled"))
   {
      m_queue_model = QueueModel::create("dram-queue", core_id, Sim()->getCfg()->getString("perf_model/dram/queue_model/type"),
                                         m_dram_bandwidth.getRoundedLatency(8 * cache_block_size)); // bytes to bits
   }

   registerStatsMetric("dram", core_id, "total-access-latency", &m_total_access_latency);
   registerStatsMetric("dram", core_id, "total-queueing-delay", &m_total_queueing_delay);
}

DramPerfModelConstant::~DramPerfModelConstant()
{
   if (m_queue_model)
   {
     delete m_queue_model;
      m_queue_model = NULL;
   }
}

SubsecondTime
DramPerfModelConstant::getAccessLatency(SubsecondTime pkt_time, UInt64 pkt_size, core_id_t requester, IntPtr address, DramCntlrInterface::access_t access_type, ShmemPerf *perf)
{
   // pkt_size is in 'Bytes'
   // m_dram_bandwidth is in 'Bits per clock cycle'
   if ((!m_enabled) ||
         (requester >= (core_id_t) Config::getSingleton()->getApplicationCores()))
   {
      return SubsecondTime::Zero();
   }

   SubsecondTime processing_time = m_dram_bandwidth.getRoundedLatency(8 * pkt_size); // bytes to bits

   // Compute Queue Delay
   SubsecondTime queue_delay;
   if (m_queue_model)
   {
      queue_delay = m_queue_model->computeQueueDelay(pkt_time, processing_time, requester);
   }
   else
   {
      queue_delay = SubsecondTime::Zero();
   }

   UInt32 bank_nr = get_address_bank(address, requester);
   int bank_mode = Sim()->m_bank_modes[bank_nr];

   SubsecondTime access_latency;

   // Distinguish between dram power modes.
   if (bank_mode == LOW_POWER) // Low power mode
   {
      access_latency = queue_delay + processing_time + m_dram_access_cost_lowpower;
   }
   else // Normal power mode.
   {
      access_latency = queue_delay + processing_time + m_dram_access_cost;
   }

   // ########################################
   // if (pkt_time.getNS() < 100){
   //    printf("pkt_time:                %lu\n", pkt_time.getNS());
   //    // printf("queue_delay(ns):         %lu\n", queue_delay.getNS());
   //    // printf("processing_time(ns):     %lu\n", processing_time.getNS());
   //    // printf("m_dram_access_cost(ns):  %lu\n", m_dram_access_cost.getNS());
   //    printf("access_latency(ns):      %lu\n\n", access_latency.getNS());
   //    // printf("acc_time:                %d\n", ACCUMULATION_TIME);
   //    // printf("start_time:              %d\n", access_interval_start_time);
   //    // printf("pkt_size:                %lu\n\n", pkt_size);
   // }


   perf->updateTime(pkt_time);
   perf->updateTime(pkt_time + queue_delay, ShmemPerf::DRAM_QUEUE);
   perf->updateTime(pkt_time + queue_delay + processing_time, ShmemPerf::DRAM_BUS);
   perf->updateTime(pkt_time + queue_delay + processing_time + m_dram_access_cost, ShmemPerf::DRAM_DEVICE);

   // Update Memory Counters
   m_num_accesses ++;
   m_total_access_latency += access_latency;
   m_total_queueing_delay += queue_delay;

   if (accumulation == 1) {
      UInt64 current_time = pkt_time.getNS();
      total_accesses[requester]++;
      total_queue_delay[requester] += queue_delay.getNS();
      total_processing_time[requester] += processing_time.getNS();
      total_m_dram_access_cost[requester] += m_dram_access_cost.getNS();
      total_access_latency[requester] += access_latency.getNS();


      if (current_time > ACCUMULATION_TIME + access_interval_start_time) {
         std::fstream info_file(ACCESS_INFO_FILE, std::ios_base::in);
         UInt32 filenumber;
         info_file >> filenumber;
         if (access_newfile == 0) {
            ofstream f ((ACCESS_INFO_FILE).c_str());
            if (f.is_open()) {
               f << ++filenumber;
            }
            ++access_newfile;
            ofstream file ((ACCESS_DATA_FOLDER + "dram_access_data_acc" + std::to_string(filenumber) + ".csv").c_str(), ios::app);
            if (file.is_open()) {
               file << ACCESS_DATA_HEADER;
            }
         }
         ofstream file ((ACCESS_DATA_FOLDER + "dram_access_data_acc" + std::to_string(filenumber) + ".csv").c_str(), ios::app);
         if (file.is_open()) {
            for (UInt32 i = 0; i < MAX_NUM_OF_CORES; i = i + 1){
               if (total_accesses[i] > 0) {
                  file   << access_interval_start_time + ACCUMULATION_TIME
                  << "," << i
                  << "," << total_accesses[i]
                  // << "," << total_queue_delay[i] / total_accesses[i]
                  // << "," << total_processing_time[i] / total_accesses[i]
                  // << "," << total_m_dram_access_cost[i] / total_accesses[i]
                  << "," << total_access_latency[i] / total_accesses[i]
                  << "\n";
               }
            }
            file.close();
         } else cout << "Unable to open file\n";
      }
      access_interval_start_time = current_time - (current_time % ACCUMULATION_TIME);
      memset(total_accesses, 0, sizeof(total_accesses));
      memset(total_queue_delay, 0, sizeof(total_queue_delay));
      memset(total_processing_time, 0, sizeof(total_processing_time));
      memset(total_m_dram_access_cost, 0, sizeof(total_m_dram_access_cost));
      memset(total_access_latency, 0, sizeof(total_access_latency));
   } else {
      std::fstream info_file(ACCESS_INFO_FILE, std::ios_base::in);
      UInt32 filenumber;
      info_file >> filenumber;
      if (access_newfile == 0) {
         ofstream f ((ACCESS_INFO_FILE).c_str());
         if (f.is_open()) {
            f << ++filenumber;
         }
         ++access_newfile;
         ofstream file ((ACCESS_DATA_FOLDER + "dram_access_data_raw" + std::to_string(filenumber) + ".csv").c_str(), ios::app);
         if (file.is_open()) {
            file << ACCESS_DATA_HEADER_RAW;
         }
      }
      ofstream file ((ACCESS_DATA_FOLDER + "dram_access_data_raw" + std::to_string(filenumber) + ".csv").c_str(), ios::app);
      if (file.is_open()) {
         file   << pkt_time.getNS()
         << "," << requester
         << "," << access_latency.getNS()
         << "\n";
         file.close();
      } else cout << "Unable to open file\n";
   }
   // ########################################

   return access_latency;
}
