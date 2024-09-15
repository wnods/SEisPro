#include <iostream>
#include <fstream>
#include <vector>

void convertDatToSegy(const std::string& datFile, const std::string& segyFile) {
    /
    std::ifstream datStream(datFile, std::ios::binary);
    if (!datStream.is_open()) {
        std::cerr << "Erro ao abrir o arquivo .dat" << std::endl;
        return;
    }

    
    std::vector<char> data((std::istreambuf_iterator<char>(datStream)), std::istreambuf_iterator<char>());
    datStream.close();

    
    std::ofstream segyStream(segyFile, std::ios::binary);
    if (!segyStream.is_open()) {
        std::cerr << "Erro ao abrir o arquivo .SEGY" << std::endl;
        return;
    }

    
    char segyHeader[3200] = {0}; 
    segyStream.write(segyHeader, sizeof(segyHeader));

    
    segyStream.write(data.data(), data.size());
    segyStream.close();

    std::cout << "Conversão concluída com sucesso!" << std::endl;
}

int main() {
    std::string datFile = "input.dat";
    std::string segyFile = "output.segy";

    convertDatToSegy(datFile, segyFile);

    return 0;
}
