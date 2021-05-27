/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

namespace ska::sdp
{
    class RecordBatch;

    class BasicProcessor
    {
        BasicProcessor(std::vector<std::string> schema, std::string plasma_path, std::string prefix, std::string name)
        {

        }
        
    public:
        virtual void process(double timeout) {}
        virtual void parameter() {}
        virtual void oid_parameter() {}
        virtual void output_tensor(RecordBatch& batch, std::string name, std::string type) {}

    protected:
        void process_call(std::string proc_func, RecordBatch& batch)
    };

    class Processor : public BasicProcessor
    {
        void process_call(std:;string proc_func, RecordBatch& batch) override
        {

        }
    };
}

int main()
{
    // SKA-SDP-DAL Processor


}