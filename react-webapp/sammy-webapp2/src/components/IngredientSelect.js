import React, {useState} from "react";
import WindowedSelect from "react-windowed-select";
import getAll from '../middle/getAll'

export default function MultiSelect() {
  const [options, setOptions] = useState([]);
  const [selectedList, setSelectedList] = useState(0);

  getAll(setOptions)

  return (
    <WindowedSelect
      options={options}
      onChange={() => setSelectedList()}
      value={selectedList}
      isMulti
      closeMenuOnSelect={false}
    />
    );
}



