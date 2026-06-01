export function triggerDownload(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

export function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    triggerDownload(url, filename);
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadText(content, filename, type = 'text/plain;charset=utf-8') {
    downloadBlob(new Blob([content], { type }), filename);
}

export const ZIP_CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i += 1) {
        let c = i;
        for (let k = 0; k < 8; k += 1) {
            c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        }
        table[i] = c >>> 0;
    }
    return table;
})();

export function crc32(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) {
        crc = ZIP_CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

export function writeZipUint16(target, offset, value) {
    target[offset] = value & 0xff;
    target[offset + 1] = (value >>> 8) & 0xff;
}

export function writeZipUint32(target, offset, value) {
    target[offset] = value & 0xff;
    target[offset + 1] = (value >>> 8) & 0xff;
    target[offset + 2] = (value >>> 16) & 0xff;
    target[offset + 3] = (value >>> 24) & 0xff;
}

export function zipDosTimestamp(date = new Date()) {
    const year = Math.max(1980, Math.min(2107, date.getFullYear()));
    return {
        time: (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2),
        date: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
    };
}

export function createZipBlob(entries, normalizeName) {
    const encoder = new TextEncoder();
    const localParts = [];
    const centralParts = [];
    const timestamp = zipDosTimestamp();
    let offset = 0;
    const usedNames = new Set();

    for (const entry of entries) {
        const name = normalizeName(entry.name, usedNames);
        const nameBytes = encoder.encode(name);
        const dataBytes = encoder.encode(entry.content || '');
        const checksum = crc32(dataBytes);

        const localHeader = new Uint8Array(30 + nameBytes.length);
        writeZipUint32(localHeader, 0, 0x04034b50);
        writeZipUint16(localHeader, 4, 20);
        writeZipUint16(localHeader, 6, 0x0800);
        writeZipUint16(localHeader, 8, 0);
        writeZipUint16(localHeader, 10, timestamp.time);
        writeZipUint16(localHeader, 12, timestamp.date);
        writeZipUint32(localHeader, 14, checksum);
        writeZipUint32(localHeader, 18, dataBytes.length);
        writeZipUint32(localHeader, 22, dataBytes.length);
        writeZipUint16(localHeader, 26, nameBytes.length);
        writeZipUint16(localHeader, 28, 0);
        localHeader.set(nameBytes, 30);
        localParts.push(localHeader, dataBytes);

        const centralHeader = new Uint8Array(46 + nameBytes.length);
        writeZipUint32(centralHeader, 0, 0x02014b50);
        writeZipUint16(centralHeader, 4, 20);
        writeZipUint16(centralHeader, 6, 20);
        writeZipUint16(centralHeader, 8, 0x0800);
        writeZipUint16(centralHeader, 10, 0);
        writeZipUint16(centralHeader, 12, timestamp.time);
        writeZipUint16(centralHeader, 14, timestamp.date);
        writeZipUint32(centralHeader, 16, checksum);
        writeZipUint32(centralHeader, 20, dataBytes.length);
        writeZipUint32(centralHeader, 24, dataBytes.length);
        writeZipUint16(centralHeader, 28, nameBytes.length);
        writeZipUint16(centralHeader, 30, 0);
        writeZipUint16(centralHeader, 32, 0);
        writeZipUint16(centralHeader, 34, 0);
        writeZipUint16(centralHeader, 36, 0);
        writeZipUint32(centralHeader, 38, 0);
        writeZipUint32(centralHeader, 42, offset);
        centralHeader.set(nameBytes, 46);
        centralParts.push(centralHeader);

        offset += localHeader.length + dataBytes.length;
        usedNames.add(name);
    }

    const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0);
    const end = new Uint8Array(22);
    writeZipUint32(end, 0, 0x06054b50);
    writeZipUint16(end, 8, entries.length);
    writeZipUint16(end, 10, entries.length);
    writeZipUint32(end, 12, centralSize);
    writeZipUint32(end, 16, offset);
    writeZipUint16(end, 20, 0);
    return new Blob([...localParts, ...centralParts, end], { type: 'application/zip' });
}
